// Tiny rules engine and UI glue (no deps)
const $ = (s) => document.querySelector(s);

const REL_WEIGHT = {
  intimate: -0.2,
  friend: 0.0,
  coworker: 0.1,
  formal: 0.2,
};

const MAP = {
  happy: {
    keywords: ["yay","great","awesome","love","喜欢","太棒了","开心","哈哈","lol","lmao","nice"],
    emojis: ["😊","😆","😀","😄","🥳","😍"]
  },
  angry: {
    keywords: ["angry","mad","wtf","damn","shit","生气","烦","妈的","tmd","为什么不"],
    emojis: ["😠","😡","🤬"]
  },
  uncertain: {
    keywords: ["maybe","perhaps","not sure","不知道","可能","吗","maybe?","idk"],
    emojis: ["🤔","😕","😐"]
  }
};

function scoreTone(text) {
  const t = text.toLowerCase();
  let s = {happy:0, angry:0, uncertain:0};

  // keywords
  for (const tone of Object.keys(MAP)) {
    for (const kw of MAP[tone].keywords) {
      const count = (t.match(new RegExp(kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), "g"))||[]).length;
      s[tone] += count * 1.0;
    }
  }
  // emojis
  for (const tone of Object.keys(MAP)) {
    for (const e of MAP[tone].emojis) {
      const count = (text.match(new RegExp(e, "g"))||[]).length;
      s[tone] += count * 1.2;
    }
  }
  // punctuation / casing
  const exclam = (text.match(/!/g)||[]).length;
  const qmarks = (text.match(/\?/g)||[]).length;
  if (exclam >= 2) s.happy += 0.8;
  if (qmarks >= 2) s.uncertain += 0.8;

  // CAPS words heuristic
  const words = text.split(/\s+/).filter(Boolean);
  const caps = words.filter(w => w.length>=3 && w === w.toUpperCase() && /[A-Z]/.test(w)).length;
  if (caps >= 1) s.angry += 0.7;

  // Choose max
  let tone = "calm";
  let maxv = 0;
  for (const k of ["happy","angry","uncertain"]) {
    if (s[k] > maxv) { maxv = s[k]; tone = k; }
  }

  // intensity normalize: rough
  const intensity = Math.max(0, Math.min(1, maxv / 3.0));
  return { tone, intensity, raw: s };
}

function applyRelation(seriousness, relation) {
  const w = REL_WEIGHT[relation] ?? 0;
  return Math.max(0, Math.min(1, seriousness + w));
}

function toneToClass(tone) {
  switch (tone) {
    case "happy": return "tone-happy";
    case "angry": return "tone-angry";
    case "uncertain": return "tone-uncertain";
    default: return "tone-calm";
  }
}

function analyzeAndRender() {
  const text = $("#msg").value;
  const relation = $("#rel").value;
  const { tone, intensity, raw } = scoreTone(text);

  // seriousness ~ tone intensity (can be refined). Apply relation.
  const seriousness = applyRelation(intensity, relation);

  // Render preview
  const preview = $("#preview");
  preview.className = "bubble " + toneToClass(tone);
  preview.style.opacity = String(0.85 + 0.15 * seriousness);
  preview.textContent = text || "Your live preview will appear here.";

  $("#meta").textContent = `tone=${tone} | intensity=${intensity.toFixed(2)} | relation=${relation} | seriousness=${seriousness.toFixed(2)}`;

  // Return payload for insertion
  return { tone, intensity, relation, seriousness, text };
}

$("#analyze").addEventListener("click", analyzeAndRender);
$("#msg").addEventListener("input", analyzeAndRender);
$("#rel").addEventListener("change", analyzeAndRender);

$("#insert").addEventListener("click", async () => {
  const payload = analyzeAndRender();
  const tag = `[${payload.tone.toUpperCase()}] `;
  const content = tag + payload.text;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      await chrome.tabs.sendMessage(tab.id, { type: "INSERT_TEXT", content });
    }
  } catch (e) {
    console.error(e);
  }
});

analyzeAndRender();
