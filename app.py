import os
import secrets
import re
import html
import yaml
import markdown
import requests
from functools import wraps
from flask import Flask, render_template, redirect, url_for, session, request, jsonify
from markupsafe import escape
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Hugging Face OAuth Configuration
HF_CLIENT_ID = os.environ.get('HF_CLIENT_ID', '')
HF_CLIENT_SECRET = os.environ.get('HF_CLIENT_SECRET', '')
HF_REDIRECT_URI = os.environ.get('HF_REDIRECT_URI', 'http://localhost:5000/callback')
HF_AUTH_URL = 'https://huggingface.co/oauth/authorize'
HF_TOKEN_URL = 'https://huggingface.co/oauth/token'
HF_API_URL = 'https://router.huggingface.co/v1/chat/completions'

# Default model for inference
DEFAULT_MODEL = os.environ.get('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')


def load_actions(example_type):
    """Load action definitions from YAML files for a specific example type."""
    actions_dir = os.path.join('actions', example_type)
    actions = {}
    
    import_file = os.path.join(actions_dir, 'import.yaml')
    if os.path.exists(import_file):
        with open(import_file, 'r', encoding='utf-8') as f:
            import_config = yaml.safe_load(f)
            if import_config and import_config.get('type') == 'import':
                for filename in import_config.get('files', []):
                    action_file = os.path.join(actions_dir, filename)
                    if os.path.exists(action_file):
                        with open(action_file, 'r', encoding='utf-8') as af:
                            action_config = yaml.safe_load(af)
                            if action_config and action_config.get('type') == 'action':
                                actions[action_config['name']] = action_config.get('definition', '')
    
    return actions


def build_system_prompt(actions):
    """Build system prompt with DFA-Notation-Grammar rules."""
    action_definitions = "\n".join([
        f"### {name}\n{definition}" 
        for name, definition in actions.items()
    ])
    
    return f"""# 선언성 자유 형식 표기 행동 문법

## 기본 원칙
당신은 아래 정의된 행동 명칭을 사용하여 선언적 행동 표기를 할 수 있습니다. 정의된 행동 명칭만 사용 가능하며, 자유 형식으로 내용을 작성할 수 있습니다.

## 문법 형식
[{{행동 명칭}}내용]

## 사용 가능한 행동 명칭
{action_definitions}

## 핵심 규칙
1. **파일 종속성**: 위에 명시된 행동 명칭만 사용
2. **내용 자유도**: 내용은 해당 행동의 정의 범위 내에서 자유 작성
3. **위치 무제한**: 응답의 시작, 중간, 끝 어디든 배치 가능
4. **사용 빈도**: 여러 번 사용하거나 생략 가능

## 주의사항
- 정의되지 않은 행동 명칭 사용 금지
- 행동 내용은 맥락과 일치해야 함
- 자연스러운 응답 흐름 유지
- none_action은 사용자만 사용 가능, 당신은 사용 금지

## 응답 구조 권장
행동 표기를 포함하되, 전체 응답이 유창하고 자연스럽도록 구성하라."""


def parse_action_tags(text, show_action_name=False):
    """Parse action tags and convert to styled HTML."""
    pattern = r'\[\{(\w+)\}([^\]]*)\]'
    
    def replace_tag(match):
        action_name = html.escape(match.group(1))
        content = html.escape(match.group(2))
        if show_action_name:
            return f'<span class="action-tag"><span class="action-name">{action_name}</span><span class="action-content">{content}</span></span>'
        else:
            return f'<span class="action-tag"><span class="action-content">{content}</span></span>'
    
    return re.sub(pattern, replace_tag, text)


def render_message(text, show_action_name=False):
    """Render message with markdown and action tags."""
    # First parse action tags
    text_with_actions = parse_action_tags(text, show_action_name)
    # Then render markdown (but preserve our action spans)
    # Simple markdown rendering for basic formatting
    html = markdown.markdown(text_with_actions, extensions=['fenced_code', 'tables'])
    return html


def extract_actions_from_response(text):
    """Extract action tags from response for avatar/robot effects."""
    pattern = r'\[\{(\w+)\}([^\]]*)\]'
    matches = re.findall(pattern, text)
    return [{'action': m[0], 'content': m[1]} for m in matches]


def login_required(f):
    """Decorator to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'access_token' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    """Main page with example links."""
    logged_in = 'access_token' in session
    username = session.get('username', '')
    return render_template('index.html', logged_in=logged_in, username=username)


@app.route('/login')
def login():
    """Initiate Hugging Face OAuth login."""
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    
    auth_url = (
        f"{HF_AUTH_URL}?"
        f"client_id={HF_CLIENT_ID}&"
        f"redirect_uri={HF_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=openid profile inference-api&"
        f"state={state}"
    )
    return redirect(auth_url)


@app.route('/callback')
def callback():
    """Handle OAuth callback from Hugging Face."""
    error = request.args.get('error')
    if error:
        return render_template('error.html', error=error), 400
    
    code = request.args.get('code')
    state = request.args.get('state')
    
    # Verify state to prevent CSRF
    if state != session.get('oauth_state'):
        return render_template('error.html', error='Invalid state parameter'), 400
    
    # Exchange code for access token
    token_data = {
        'client_id': HF_CLIENT_ID,
        'client_secret': HF_CLIENT_SECRET,
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': HF_REDIRECT_URI
    }
    
    try:
        response = requests.post(HF_TOKEN_URL, data=token_data)
        response.raise_for_status()
        token_response = response.json()
        
        session['access_token'] = token_response.get('access_token')
        session['refresh_token'] = token_response.get('refresh_token')
        
        # Get user info
        user_response = requests.get(
            'https://huggingface.co/api/whoami-v2',
            headers={'Authorization': f"Bearer {session['access_token']}"}
        )
        if user_response.ok:
            user_data = user_response.json()
            session['username'] = user_data.get('name', user_data.get('fullname', 'User'))
        
        session.pop('oauth_state', None)
        return redirect(url_for('index'))
        
    except requests.RequestException as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/logout')
def logout():
    """Clear session and logout."""
    session.clear()
    return redirect(url_for('index'))


@app.route('/chatbot')
@login_required
def chatbot():
    """Chatbot example page."""
    actions = load_actions('chatbot')
    return render_template('chatbot.html', actions=actions, username=session.get('username', ''), logged_in=True)


@app.route('/avatar')
@login_required
def avatar():
    """Avatar example page."""
    actions = load_actions('avatar')
    return render_template('avatar.html', actions=actions, username=session.get('username', ''), logged_in=True)


@app.route('/robot')
@login_required
def robot():
    """Robot example page."""
    actions = load_actions('robot')
    return render_template('robot.html', actions=actions, username=session.get('username', ''), logged_in=True)


@app.route('/affinity')
@login_required
def affinity():
    """Affinity/Favorability example page."""
    actions = load_actions('affinity')
    return render_template('affinity.html', actions=actions, username=session.get('username', ''), logged_in=True)


@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """API endpoint for chat with AI."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    user_message = data.get('message', '')
    example_type = data.get('example_type', 'chatbot')
    conversation_history = data.get('history', [])
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Load actions for the example type
    actions = load_actions(example_type)
    system_prompt = build_system_prompt(actions)
    
    # Build conversation for the model (OpenAI-compatible format)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation_history[-10:]:  # Keep last 10 messages for context
        messages.append({"role": msg['role'], "content": msg['content']})
    messages.append({"role": "user", "content": user_message})
    
    # Call Hugging Face Inference API (OpenAI-compatible endpoint)
    headers = {
        'Authorization': f"Bearer {session['access_token']}",
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': DEFAULT_MODEL,
        'messages': messages,
        'max_tokens': 500,
        'temperature': 0.7,
        'top_p': 0.9,
        'stream': False
    }
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 401:
            session.clear()
            return jsonify({'error': 'Token expired. Please login again.', 'redirect': '/login'}), 401
        
        response.raise_for_status()
        result = response.json()
        
        # Parse OpenAI-compatible response format
        if 'choices' in result and len(result['choices']) > 0:
            ai_response = result['choices'][0].get('message', {}).get('content', '')
        else:
            ai_response = str(result)
        
        # Clean up the response
        ai_response = ai_response.strip()
        
        # Extract actions for effects
        extracted_actions = extract_actions_from_response(ai_response)
        
        # Determine if we should show action names (robot, affinity show them)
        show_action_name = example_type in ['robot', 'affinity']
        
        # Render the response with markdown and action tags
        rendered_response = render_message(ai_response, show_action_name)
        
        # Render user message as well (for action tag styling)
        user_rendered = render_message(user_message, show_action_name)
        
        return jsonify({
            'response': ai_response,
            'rendered': rendered_response,
            'user_rendered': user_rendered,
            'actions': extracted_actions
        })
        
    except requests.Timeout:
        return jsonify({'error': 'Request timed out. The model might be loading.'}), 504
    except requests.RequestException as e:
        return jsonify({'error': f'API error: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
