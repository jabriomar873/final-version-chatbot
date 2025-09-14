css = '''
<style>
/* Minimal, clean layout */
.chat-message {
    display: flex;
    gap: 12px;
    padding: 12px 14px;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #ffffff;
    margin: 8px 0;
    position: relative; /* ensure above watermark */
    z-index: 1;
}
.chat-message.user { border-left: 3px solid #3b82f6; }
.chat-message.bot  { border-left: 3px solid #10b981; }

.chat-message .avatar { width: 36px; height: 36px; }
.chat-message .avatar img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    flex: 1;
    color: #111827;
    font: 400 14px/1.55 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    white-space: pre-wrap;
}

/* Citation highlight */
.chat-message .message span.citation {
    background: #ecfdf5; /* light emerald */
    border: 1px solid #10b981;
    padding: 0 4px;
    margin: 0 2px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    color: #065f46;
    display: inline-block;
}

.stForm { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #fff; }

/* Subtle background watermark (hidden brand) */
.stApp:before {
    content: "CAVEO";
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%); /* horizontal now */
    font-size: 18vw; /* slightly smaller when horizontal */
    font-weight: 800;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    letter-spacing: 0.6vw; /* tighter for horizontal */
    color: rgba(0,0,0,0.035); /* keep very light */
    pointer-events: none;
    user-select: none;
    z-index: 0;
    white-space: nowrap;
}

/* Dark mode safeguard (if user theme flips) */
@media (prefers-color-scheme: dark) {
    .stApp:before { color: rgba(255,255,255,0.04); }
}

/* Green action buttons (sidebar) */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button {
    background: #059669 !important; /* emerald-600 */
    border: 1px solid #047857 !important;
    color: #ffffff !important;
    font-weight: 600;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    transition: background .15s ease, transform .15s ease;
    min-height: 46px;
    width: 100%;
    font-size: 15px;
}
section[data-testid="stSidebar"] button:hover {
    background: #10b981 !important; /* emerald-500 */
}
section[data-testid="stSidebar"] button:active {
    transform: translateY(1px);
    background: #047857 !important;
}

/* File uploader as green panel */
section[data-testid="stSidebar"] .st-file-uploader {
    border: 1px dashed #047857 !important;
    background: rgba(16,185,129,0.08);
    border-radius: 8px;
    padding: 6px 6px 2px 6px;
}
section[data-testid="stSidebar"] .st-file-uploader div[role="button"],
section[data-testid="stSidebar"] .st-file-uploader label {
    color: #059669 !important;
    font-weight: 600;
}
/* Hide Streamlit's default helper texts (drag & drop + size limit) */
section[data-testid="stSidebar"] .stFileUploader label + div small,
section[data-testid="stSidebar"] .stFileUploader small,
section[data-testid="stSidebar"] .st-file-uploader small,
section[data-testid="stSidebar"] .stFileUploader span:has(svg) + div { display:none !important; }

/* Hide Streamlit chrome: menu, deploy, rerun, footer, toolbar */
header[data-testid="stHeader"] div:nth-child(2), /* toolbar elements */
header [data-testid="stToolbar"],
header button[kind="header"],
div[data-testid="stStatusWidget"],
button[title="View fullscreen"],
section[data-testid="stSidebar"] [data-testid="stThemeSwitcher"],
div.viewerBadge_container__1QSob, /* footer badge */
footer { display: none !important; }

/* Remove gap left by hidden header */
header[data-testid="stHeader"] { height: 0px !important; }
header[data-testid="stHeader"] > * { display:none; }
main.block-container { padding-top: 1rem !important; }
</style>
'''

# Base64 fallback tiny placeholder (replace with real caveo.jpg if present)
# Inline ChatGPT-like swirl SVG (white logo on emerald background)
# Themeable background color for bot icon
_BOT_ICON_BG = '#374151'  # Slate gray; adjust here for different theme

_CHATGPT_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100' width='36' height='36'>"
    f"<rect width='100' height='100' rx='18' fill='{_BOT_ICON_BG}'/>"
    "<path fill='white' d='M50 22c6.5 0 12.4 3.2 16 8.1 5.7 1 10 6 10 12.1 0 2.4-.7 4.7-2 6.6.5 1.7.8 3.4.8 5.2 0 11.6-9.4 21-21 21-5.7 0-10.9-2.3-14.7-6-1.3.4-2.8.6-4.3.6-8 0-14.5-6.5-14.5-14.5 0-2.4.6-4.6 1.6-6.6C20.6 56.1 20 54 20 51.8c0-6.8 4.8-12.5 11.2-13.9C34 30 41.2 22 50 22Zm0 6c-6.1 0-11.3 4.2-12.7 10.1l-.6 2.5-2.6.4c-4.3.7-7.6 4.4-7.6 8.8 0 1.5.4 3 1 4.3l1.3 2.5-1.5 2.4c-.8 1.3-1.2 2.8-1.2 4.3 0 4.7 3.8 8.5 8.5 8.5 1.3 0 2.6-.3 3.7-.8l2.6-1.3 2 2.1c2.8 3 6.6 4.7 10.5 4.7 8.3 0 15-6.7 15-15 0-1.4-.2-2.8-.6-4.1l-.8-2.7 1.7-2.3c1-1.3 1.5-2.9 1.5-4.5 0-4.2-3-7.7-7.1-8.4l-2.6-.5-.9-2.5C60.9 34.9 55.8 30 50 30Z'/>"
    "</svg>"
)

bot_template = (
    '<div class="chat-message bot">'
    '<div class="avatar" style="display:flex;align-items:center;justify-content:center;width:36px;height:36px;">'
    + _CHATGPT_SVG +
    '</div>'
    '<div class="message">__MSG__</div>'
    '</div>'
)

user_template = (
    '<div class="chat-message user">'
    '<div class="message" style="margin-left:4px;">__MSG__</div>'
    '</div>'
)