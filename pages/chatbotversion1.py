import streamlit as st
import requests
import uuid
import webbrowser

# Cấu hình trang
st.set_page_config(
    page_title="Trà Toàn Thắng Chatbot",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Áp dụng CSS tùy chỉnh
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
            font-family: Arial, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        h1, h2, h3 {
            color: #1b5e20;
        }
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #1b5e20;
        }
        .chat-message {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 5px;
        }
        .user-message {
            background-color: #dcedc8;
        }
        .assistant-message {
            background-color: #f9fbe7;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image("https://tratoanthang.com/uploads/source/up/logo.png", caption="Trà Toàn Thắng", use_container_width=True)
    st.markdown("## 🌟 Đặt hàng trực tuyến tại Website của chúng tôi")
    link_page = 'https://tratoanthang.com/'
    if st.button("🌏 Truy cập Website"):
        webbrowser.open_new_tab(link_page)

# Giao diện chính
# st.image("https://tratoanthang.com/uploads/source/up/logo.png", use_container_width=True)
st.title("🛍️ Trà Toàn Thắng Chatbot")
st.subheader("Chào mừng bạn đến với cửa hàng của chúng tôi")
st.markdown("💡 **Hỗ trợ tư vấn 24/7 và đặt hàng nhanh chóng qua Chatbot.**")

# API setup
try:
    # ngrok_url = f.read().strip()
    ngrok_url = r"https://e207-42-118-114-81.ngrok-free.app"
    st.session_state.flask_api_url_1 = ngrok_url + "/v2/chat"
except FileNotFoundError:
    st.error("Error: ngrok_url.txt not found. Please run app.py first.")
    st.stop()
session_id = str(uuid.uuid4())

# Initialize chat history
if "chat_history_v1" not in st.session_state:
    st.session_state.chat_history_v1 = []  # Corrected initialization

# Display chat
st.markdown("---")
for message in st.session_state.chat_history_v1:
    css_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"<div class='chat-message {css_class}'>{message['content']}</div>", unsafe_allow_html=True)

# User input
if prompt := st.chat_input(key="chat", placeholder="Hãy nhập câu hỏi của bạn tại đây..."):
    st.session_state.chat_history_v1.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "message": {"human": prompt},
        "context": st.session_state.chat_history_v1,
        "sessionId": session_id,
        "stream": True
    }

    with st.chat_message("assistant"):
        streamed_content = ""
        response = requests.post(st.session_state.flask_api_url_1, json=payload, stream=True)
        response_placeholder = st.empty()
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    streamed_content += chunk
                    response_placeholder.markdown(streamed_content)
            st.session_state.chat_history_v1.append({"role": "assistant", "content": streamed_content})
        else:
            st.error(f"Error: {response.status_code}")