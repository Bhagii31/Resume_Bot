/* ─────────────────────────────────────────────────────────────
   Resume Assistant — frontend logic
   Talks to the FastAPI /api/chat endpoint and renders the streamed
   answer token-by-token.
   ───────────────────────────────────────────────────────────── */

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const welcome = document.getElementById("welcome");

// Auto-grow the textarea as you type.
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

// Enter sends; Shift+Enter makes a new line.
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// Suggested-question chips fill the input and send.
document.getElementById("chips").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    input.value = e.target.textContent;
    form.requestSubmit();
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (question) ask(question);
});

/** Append a message bubble and return its inner text node for streaming. */
function addMessage(role, text) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

/** Show the three-dot typing indicator inside a fresh bot bubble. */
function addTyping() {
  const bubble = addMessage("bot", "");
  bubble.innerHTML =
    '<span class="typing"><span></span><span></span><span></span></span>';
  return bubble;
}

async function ask(question) {
  // Remove the welcome panel on first message.
  if (welcome) welcome.remove();

  // Lock the composer while we wait.
  input.value = "";
  input.style.height = "auto";
  input.disabled = true;
  sendBtn.disabled = true;

  addMessage("user", question);
  const bubble = addTyping();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`Server responded ${resp.status}`);
    }

    // Stream the plain-text response and render as it arrives.
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let answer = "";
    let first = true;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      answer += decoder.decode(value, { stream: true });
      if (first) {
        bubble.textContent = ""; // clear the typing dots
        first = false;
      }
      bubble.textContent = answer;
      chat.scrollTop = chat.scrollHeight;
    }

    if (!answer.trim()) bubble.textContent = "(no answer returned)";
  } catch (err) {
    bubble.textContent = `⚠️ Couldn't reach the server: ${err.message}`;
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

input.focus();
