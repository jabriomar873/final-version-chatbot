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

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 100 100'><circle cx='50' cy='50' r='48' fill='%2310b981'/><path d='M50 18 32 26v18l18 8 18-8V26L50 18zm0 30-18-8v24l18 12 18-12V40l-18 8z' fill='white' stroke='white' stroke-width='2' stroke-linejoin='round'/></svg>" alt="assistant">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar" style="background:transparent;width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;color:#3b82f6;font-weight:600;">U</div>
    <div class="message">{{MSG}}</div>
</div>
'''