// Content script: insert incoming text into the focused input/textarea or contenteditable element.
function insertAtCursor(el, text) {
  if (!el) return false;
  if (el.isContentEditable) {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return false;
    const range = sel.getRangeAt(0);
    range.deleteContents();
    range.insertNode(document.createTextNode(text));
    range.collapse(false);
    return true;
  }
  if ("value" in el) {
    const start = el.selectionStart ?? el.value.length;
    const end = el.selectionEnd ?? el.value.length;
    const before = el.value.slice(0, start);
    const after = el.value.slice(end);
    el.value = before + text + after;
    const pos = start + text.length;
    el.setSelectionRange?.(pos, pos);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  }
  return false;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === "INSERT_TEXT") {
    const active = document.activeElement;
    const ok = insertAtCursor(active, msg.content);
    sendResponse?.({ ok });
  }
});
