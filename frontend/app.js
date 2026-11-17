/**
 * DocuSense Enterprise RAG - Client Application
 */

// Detect API host (defaults to FastAPI backend at port 8000 when opened via Live Server, otherwise uses current origin on Vercel/production)
const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const API_BASE = (isLocalhost && window.location.port !== "8000")
    ? "http://127.0.0.1:8000"
    : window.location.origin;

// DOM Elements
const chatViewport = document.getElementById("chatViewport");
const chatContainer = document.getElementById("chatContainer");
const welcomeHero = document.getElementById("welcomeHero");
const questionInput = document.getElementById("questionInput");
const composerForm = document.getElementById("composerForm");
const sendBtn = document.getElementById("sendBtn");
const composerAttachBtn = document.getElementById("composerAttachBtn");
const browseFilesBtn = document.getElementById("browseFilesBtn");
const fileInput = document.getElementById("fileInput");
const uploadDropzone = document.getElementById("uploadDropzone");
const uploadProgressContainer = document.getElementById("uploadProgressContainer");
const uploadProgressStatus = document.getElementById("uploadProgressStatus");
const documentList = document.getElementById("documentList");
const docCount = document.getElementById("docCount");
const totalChunksBadge = document.getElementById("totalChunksBadge");
const clearAllDocsBtn = document.getElementById("clearAllDocsBtn");
const newChatBtn = document.getElementById("newChatBtn");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const sidebarModelBadge = document.getElementById("sidebarModelBadge");
const toast = document.getElementById("toast");

// Settings Modal Elements
const settingsModalBtn = document.getElementById("settingsModalBtn");
const settingsModal = document.getElementById("settingsModal");
const closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
const cancelSettingsBtn = document.getElementById("cancelSettingsBtn");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const settingsProvider = document.getElementById("settingsProvider");
const settingsOpenAIKey = document.getElementById("settingsOpenAIKey");
const settingsGroqKey = document.getElementById("settingsGroqKey");
const settingsGeminiKey = document.getElementById("settingsGeminiKey");
const settingsModel = document.getElementById("settingsModel");
const settingsTopK = document.getElementById("settingsTopK");
const settingsChunkSize = document.getElementById("settingsChunkSize");
const settingsChunkOverlap = document.getElementById("settingsChunkOverlap");
const openaiKeyStatus = document.getElementById("openaiKeyStatus");
const groqKeyStatus = document.getElementById("groqKeyStatus");
const geminiKeyStatus = document.getElementById("geminiKeyStatus");
const toggleOpenAIEye = document.getElementById("toggleOpenAIEye");
const toggleGroqEye = document.getElementById("toggleGroqEye");
const toggleGeminiEye = document.getElementById("toggleGeminiEye");

// State
let isGenerating = false;
let indexedDocuments = [];
let totalChunks = 0;

/* ==========================================================================
   Initialization
   ========================================================================== */
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    fetchHealthAndSettings();
    fetchDocuments();
    setupEventListeners();
    setupStarterPrompts();
});

function initTheme() {
    const savedTheme = localStorage.getItem("docusense_theme") || "dark";
    document.body.dataset.theme = savedTheme;
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const current = document.body.dataset.theme;
    const next = current === "light" ? "dark" : "light";
    document.body.dataset.theme = next;
    localStorage.setItem("docusense_theme", next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    if (!themeToggleBtn) return;
    const icon = themeToggleBtn.querySelector("i");
    if (icon) {
        icon.className = theme === "light" ? "fa-solid fa-moon" : "fa-solid fa-sun";
    }
}

function showToast(message, duration = 3000) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, duration);
}

/* ==========================================================================
   Data Fetching & Document Management
   ========================================================================== */
async function fetchHealthAndSettings() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            const data = await res.json();
            totalChunks = data.total_chunks || 0;
            updateStatsBadges();
        }

        const settingsRes = await fetch(`${API_BASE}/settings`);
        if (settingsRes.ok) {
            const s = await settingsRes.json();
            if (sidebarModelBadge) {
                sidebarModelBadge.textContent = s.model || "gpt-4o-mini";
            }
        }
    } catch (err) {
        console.warn("Backend not reached yet:", err);
    }
}

async function fetchDocuments() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        if (!res.ok) throw new Error("Failed to load documents");
        const data = await res.json();
        indexedDocuments = data.documents || [];
        totalChunks = data.total_chunks || 0;
        renderDocumentList();
        updateStatsBadges();
    } catch (err) {
        console.error("Error loading documents:", err);
    }
}

function updateStatsBadges() {
    if (docCount) docCount.textContent = indexedDocuments.length;
    if (totalChunksBadge) totalChunksBadge.textContent = `${totalChunks} chunks`;
}

function getFileIconClass(filename) {
    const ext = filename.split(".").pop().toLowerCase();
    switch (ext) {
        case "pdf": return "fa-solid fa-file-pdf doc-icon pdf";
        case "docx":
        case "doc": return "fa-solid fa-file-word doc-icon word";
        case "pptx":
        case "ppt": return "fa-solid fa-file-powerpoint doc-icon slide";
        case "csv": return "fa-solid fa-file-csv doc-icon sheet";
        case "md":
        case "txt": return "fa-solid fa-file-lines doc-icon text";
        default: return "fa-solid fa-file doc-icon";
    }
}

function renderDocumentList() {
    if (!documentList) return;
    documentList.innerHTML = "";

    if (indexedDocuments.length === 0) {
        documentList.innerHTML = `
            <li class="empty-docs-state">
                <i class="fa-regular fa-file-lines"></i>
                <p>No documents uploaded yet.<br>Drop files above to index.</p>
            </li>
        `;
        return;
    }

    indexedDocuments.forEach((doc) => {
        const li = document.createElement("li");
        li.className = "document-item";
        const iconClass = getFileIconClass(doc.filename);

        li.innerHTML = `
            <div class="doc-info-group">
                <i class="${iconClass}"></i>
                <div class="doc-meta">
                    <span class="doc-filename" title="${doc.filename}">${doc.filename}</span>
                    <span class="doc-stats">${doc.chunks} chunks · ${doc.size_formatted || "Ready"}</span>
                </div>
            </div>
            <button class="btn-doc-delete" title="Delete document from knowledge base" data-filename="${doc.filename}">
                <i class="fa-regular fa-trash-can"></i>
            </button>
        `;

        const deleteBtn = li.querySelector(".btn-doc-delete");
        deleteBtn.addEventListener("click", () => deleteDocument(doc.filename));
        documentList.appendChild(li);
    });
}

async function deleteDocument(filename) {
    if (!confirm(`Remove "${filename}" from the knowledge base? All corresponding vectors will be deleted.`)) {
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
            method: "DELETE"
        });
        if (!res.ok) throw new Error("Failed to delete document");
        showToast(`Removed "${filename}"`);
        await fetchDocuments();
    } catch (err) {
        showToast(`❌ Error: ${err.message}`);
    }
}

async function clearKnowledgeBase() {
    if (!confirm("Are you sure you want to clear ALL documents and vectors from the knowledge base?")) {
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/clear`, { method: "POST" });
        if (!res.ok) throw new Error("Failed to clear knowledge base");
        showToast("Knowledge base cleared");
        await fetchDocuments();
    } catch (err) {
        showToast(`❌ Error: ${err.message}`);
    }
}

/* ==========================================================================
   File Upload Handling
   ========================================================================== */
async function handleFilesUpload(files) {
    if (!files || files.length === 0) return;

    const allowed = [".pdf", ".docx", ".txt", ".pptx", ".md", ".csv"];
    const validFiles = Array.from(files).filter(f => {
        const ext = `.${f.name.split('.').pop().toLowerCase()}`;
        return allowed.includes(ext);
    });

    if (validFiles.length === 0) {
        showToast("❌ Please upload supported files (.pdf, .docx, .txt, .pptx, .md, .csv)");
        return;
    }

    if (uploadProgressContainer) uploadProgressContainer.style.display = "flex";
    if (uploadProgressStatus) uploadProgressStatus.textContent = `Processing & chunking ${validFiles.length} file(s)...`;

    const formData = new FormData();
    validFiles.forEach(f => formData.append("files", f));

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Upload failed");
        }

        showToast(`✅ Successfully indexed ${validFiles.length} document(s)!`);
        await fetchDocuments();

        // If chat is open, announce upload completion
        addBotMessage(`📄 **Indexed ${validFiles.length} new document(s)** into knowledge base.\n\nYou can now ask specific questions, request summaries, or compare findings.`);
    } catch (err) {
        showToast(`❌ Upload error: ${err.message}`);
    } finally {
        if (uploadProgressContainer) uploadProgressContainer.style.display = "none";
        if (fileInput) fileInput.value = "";
    }
}

/* ==========================================================================
   Chat Interactions & Citations
   ========================================================================== */
function setupStarterPrompts() {
    const chips = document.querySelectorAll(".prompt-chip");
    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            const prompt = chip.getAttribute("data-prompt");
            if (prompt) {
                questionInput.value = prompt;
                handleUserSend();
            }
        });
    });
}

function handleInputKeydown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleUserSend();
    }
}

function autoResizeTextarea() {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
    sendBtn.disabled = !questionInput.value.trim() || isGenerating;
}

async function handleUserSend() {
    const query = questionInput.value.trim();
    if (!query || isGenerating) return;

    // Reset input
    questionInput.value = "";
    questionInput.style.height = "auto";
    sendBtn.disabled = true;

    // Remove welcome hero if present
    if (welcomeHero) welcomeHero.remove();

    // Render User Message
    addUserMessage(query);

    // Render Typing indicator
    const typingBubble = showTypingIndicator();
    isGenerating = true;

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: query })
        });

        const data = await res.json();
        typingBubble.remove();
        isGenerating = false;
        sendBtn.disabled = false;

        if (!res.ok) {
            throw new Error(data.detail || "Unable to get an answer.");
        }

        addBotMessage(data.answer, data.sources || [], data.notice);
    } catch (err) {
        typingBubble.remove();
        isGenerating = false;
        sendBtn.disabled = false;
        addBotMessage(`❌ **Request Error**: ${err.message}`);
    }
}

function addUserMessage(text) {
    const msg = document.createElement("div");
    msg.className = "message user";
    msg.innerHTML = `
        <div class="message-body">
            <div class="message-content">
                ${escapeHTML(text)}
            </div>
        </div>
        <div class="message-avatar">
            <i class="fa-regular fa-user"></i>
        </div>
    `;
    chatContainer.appendChild(msg);
    scrollToBottom();
}

function addBotMessage(markdownText, sources = [], notice = null) {
    const msg = document.createElement("div");
    msg.className = "message bot";

    const parsedHTML = marked.parse(markdownText);
    const msgId = "msg_" + Math.random().toString(36).substring(2, 9);

    let sourcesAccordionHTML = "";
    if (sources && sources.length > 0) {
        let cardsHTML = "";
        sources.forEach((s, idx) => {
            cardsHTML += `
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-name"><i class="fa-regular fa-file-lines"></i> ${escapeHTML(s.source)} (Page ${s.page || 1})</span>
                        <span class="confidence-badge">${s.confidence || 85}% match</span>
                    </div>
                    <div class="source-snippet">${escapeHTML(s.snippet)}</div>
                </div>
            `;
        });

        sourcesAccordionHTML = `
            <div class="sources-accordion">
                <button type="button" class="sources-toggle-btn" onclick="toggleSources('${msgId}')">
                    <span><i class="fa-solid fa-quote-left"></i> Referenced Sources (${sources.length})</span>
                    <i class="fa-solid fa-chevron-down" id="arrow_${msgId}"></i>
                </button>
                <div class="sources-content" id="content_${msgId}" style="display: none;">
                    ${cardsHTML}
                </div>
            </div>
        `;
    }

    msg.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-brain"></i>
        </div>
        <div class="message-body">
            <div class="message-content" id="text_${msgId}">
                ${parsedHTML}
            </div>
            ${sourcesAccordionHTML}
            <div class="message-actions">
                <button class="btn-message-action" onclick="copyMessageText('text_${msgId}')" title="Copy answer">
                    <i class="fa-regular fa-copy"></i> Copy
                </button>
            </div>
        </div>
    `;

    chatContainer.appendChild(msg);
    scrollToBottom();
}

function showTypingIndicator() {
    const bubble = document.createElement("div");
    bubble.className = "message bot typing";
    bubble.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-brain"></i>
        </div>
        <div class="message-body">
            <div class="typing-bubble">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    chatContainer.appendChild(bubble);
    scrollToBottom();
    return bubble;
}

function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
}

window.toggleSources = function(msgId) {
    const content = document.getElementById(`content_${msgId}`);
    const arrow = document.getElementById(`arrow_${msgId}`);
    if (!content) return;

    if (content.style.display === "none") {
        content.style.display = "flex";
        if (arrow) arrow.className = "fa-solid fa-chevron-up";
    } else {
        content.style.display = "none";
        if (arrow) arrow.className = "fa-solid fa-chevron-down";
    }
};

window.copyMessageText = function(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    navigator.clipboard.writeText(el.innerText).then(() => {
        showToast("Copied answer to clipboard!");
    }).catch(() => {
        showToast("Failed to copy");
    });
};

function escapeHTML(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/* ==========================================================================
   Settings Modal Management
   ========================================================================== */
async function openSettingsModal() {
    settingsModal.style.display = "flex";
    try {
        const res = await fetch(`${API_BASE}/settings`);
        if (!res.ok) return;
        const s = await res.json();
        settingsProvider.value = s.provider || "auto";
        settingsModel.value = s.model || "gpt-4o-mini";
        settingsTopK.value = s.top_k || 4;
        settingsChunkSize.value = s.chunk_size || 500;
        settingsChunkOverlap.value = s.chunk_overlap || 100;

        if (s.has_openai_key) {
            openaiKeyStatus.textContent = `Configured (${s.masked_openai_key})`;
            openaiKeyStatus.style.color = "var(--accent-emerald)";
        } else {
            openaiKeyStatus.textContent = "Not configured (Requires active billing at OpenAI)";
            openaiKeyStatus.style.color = "var(--text-muted)";
        }

        if (s.has_groq_key && groqKeyStatus) {
            groqKeyStatus.textContent = `Configured (${s.masked_groq_key})`;
            groqKeyStatus.style.color = "var(--accent-emerald)";
        }

        if (s.has_gemini_key && geminiKeyStatus) {
            geminiKeyStatus.textContent = `Configured (${s.masked_gemini_key})`;
            geminiKeyStatus.style.color = "var(--accent-emerald)";
        }
    } catch (e) {
        console.error("Failed to load settings:", e);
    }
}

function closeSettingsModal() {
    settingsModal.style.display = "none";
}

async function saveSettings() {
    const payload = {
        provider: settingsProvider.value,
        model: settingsModel.value,
        top_k: parseInt(settingsTopK.value, 10) || 4,
        chunk_size: parseInt(settingsChunkSize.value, 10) || 500,
        chunk_overlap: parseInt(settingsChunkOverlap.value, 10) || 100
    };

    if (settingsOpenAIKey.value.trim()) {
        payload.openai_api_key = settingsOpenAIKey.value.trim();
    }
    if (settingsGroqKey.value.trim()) {
        payload.groq_api_key = settingsGroqKey.value.trim();
    }
    if (settingsGeminiKey && settingsGeminiKey.value.trim()) {
        payload.gemini_api_key = settingsGeminiKey.value.trim();
    }

    try {
        const res = await fetch(`${API_BASE}/settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Could not save settings");
        showToast("Settings saved successfully!");
        closeSettingsModal();
        fetchHealthAndSettings();
    } catch (err) {
        showToast(`❌ Error: ${err.message}`);
    }
}

/* ==========================================================================
   Event Listeners Wire-up
   ========================================================================== */
function setupEventListeners() {
    themeToggleBtn?.addEventListener("click", toggleTheme);

    // Textarea input & resizing
    questionInput?.addEventListener("input", autoResizeTextarea);
    questionInput?.addEventListener("keydown", handleInputKeydown);
    composerForm?.addEventListener("submit", (e) => {
        e.preventDefault();
        handleUserSend();
    });

    // Upload buttons
    browseFilesBtn?.addEventListener("click", () => fileInput?.click());
    composerAttachBtn?.addEventListener("click", () => fileInput?.click());
    fileInput?.addEventListener("change", (e) => handleFilesUpload(e.target.files));

    // Drag and Drop
    if (uploadDropzone) {
        ["dragenter", "dragover"].forEach(evt => {
            uploadDropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                uploadDropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            uploadDropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                uploadDropzone.classList.remove("dragover");
            });
        });

        uploadDropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length) {
                handleFilesUpload(dt.files);
            }
        });
    }

    // Knowledge base clearing
    clearAllDocsBtn?.addEventListener("click", clearKnowledgeBase);

    // New Chat
    newChatBtn?.addEventListener("click", () => {
        chatContainer.innerHTML = "";
        const welcomeCopy = document.createElement("div");
        welcomeCopy.className = "welcome-hero";
        welcomeCopy.id = "welcomeHero";
        welcomeCopy.innerHTML = `
            <div class="hero-badge"><i class="fa-solid fa-sparkles"></i> Grounded Retrieval-Augmented Generation</div>
            <h2>New Conversation Started</h2>
            <p class="hero-description">Ask any question grounded in your ${indexedDocuments.length} indexed documents.</p>
            <div class="starter-prompts-section">
                <div class="prompt-chips">
                    <button class="prompt-chip" data-prompt="Give me an executive summary of the indexed documents.">
                        <i class="fa-regular fa-compass"></i> Provide an executive summary
                    </button>
                    <button class="prompt-chip" data-prompt="What are the key findings and conclusions?">
                        <i class="fa-regular fa-lightbulb"></i> What are the key findings?
                    </button>
                </div>
            </div>
        `;
        chatContainer.appendChild(welcomeCopy);
        setupStarterPrompts();
        showToast("New chat started");
    });

    // Settings Modal
    settingsModalBtn?.addEventListener("click", openSettingsModal);
    closeSettingsModalBtn?.addEventListener("click", closeSettingsModal);
    cancelSettingsBtn?.addEventListener("click", closeSettingsModal);
    saveSettingsBtn?.addEventListener("click", saveSettings);

    // Password visibility toggles
    toggleOpenAIEye?.addEventListener("click", () => {
        const type = settingsOpenAIKey.getAttribute("type") === "password" ? "text" : "password";
        settingsOpenAIKey.setAttribute("type", type);
        toggleOpenAIEye.innerHTML = type === "password" ? '<i class="fa-regular fa-eye"></i>' : '<i class="fa-regular fa-eye-slash"></i>';
    });

    toggleGroqEye?.addEventListener("click", () => {
        const type = settingsGroqKey.getAttribute("type") === "password" ? "text" : "password";
        settingsGroqKey.setAttribute("type", type);
        toggleGroqEye.innerHTML = type === "password" ? '<i class="fa-regular fa-eye"></i>' : '<i class="fa-regular fa-eye-slash"></i>';
    });

    toggleGeminiEye?.addEventListener("click", () => {
        const type = settingsGeminiKey.getAttribute("type") === "password" ? "text" : "password";
        settingsGeminiKey.setAttribute("type", type);
        toggleGeminiEye.innerHTML = type === "password" ? '<i class="fa-regular fa-eye"></i>' : '<i class="fa-regular fa-eye-slash"></i>';
    });
}
