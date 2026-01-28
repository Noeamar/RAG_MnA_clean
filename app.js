const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

function setActiveTab(targetId) {
  tabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.target === targetId);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === targetId);
  });
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.target));
});

document.querySelectorAll("[data-target-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    setActiveTab(button.dataset.targetTab);
  });
});

async function apiPost(path, payload) {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Erreur API");
  }
  return res;
}

function setOutput(id, value) {
  const el = document.getElementById(id);
  el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function healthCheck() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    setOutput("health-status", `Statut API: ${data.status}`);
  } catch {
    setOutput("health-status", "Statut API: indisponible");
  }
}

healthCheck();

document.getElementById("news-generate").addEventListener("click", async () => {
  const question = document.getElementById("news-question").value.trim();
  if (!question) return setOutput("news-output", "Merci de poser une question.");
  setOutput("news-output", "Chargement...");
  try {
    const res = await apiPost("/news", { question });
    const data = await res.json();
    setOutput("news-output", data.answer);
  } catch (err) {
    setOutput("news-output", err.message);
  }
});

document.getElementById("news-generate-web").addEventListener("click", async () => {
  const question = document.getElementById("news-question").value.trim();
  if (!question) return setOutput("news-output", "Merci de poser une question.");
  setOutput("news-output", "Chargement...");
  try {
    const res = await apiPost("/news-web", { question });
    const data = await res.json();
    setOutput("news-output", data.answer);
  } catch (err) {
    setOutput("news-output", err.message);
  }
});

document.getElementById("press-fetch").addEventListener("click", async () => {
  const company = document.getElementById("press-company").value.trim();
  const start_year = parseInt(document.getElementById("press-start").value, 10);
  const end_year = parseInt(document.getElementById("press-end").value, 10);
  const min_links = parseInt(document.getElementById("press-min").value, 10);
  if (!company) return setOutput("press-output", "Merci de renseigner une entreprise.");
  setOutput("press-output", "Chargement...");
  try {
    const res = await apiPost("/press-links", {
      company,
      start_year,
      end_year,
      min_links,
    });
    const data = await res.json();
    setOutput("press-output", data.links_by_year);
  } catch (err) {
    setOutput("press-output", err.message);
  }
});

document.getElementById("company-docx").addEventListener("click", async () => {
  const company_name = document.getElementById("company-name").value.trim();
  if (!company_name) return setOutput("company-output", "Merci de renseigner un nom.");
  setOutput("company-output", "Génération du Word (base interne)...");
  try {
    const res = await apiPost("/company-docx", { company_name, use_web_search: false });
    const blob = await res.blob();
    downloadBlob(blob, `${company_name}_fiche_societe.docx`.replace(/\s+/g, "_"));
    setOutput("company-output", "Fichier Word généré.");
  } catch (err) {
    setOutput("company-output", err.message);
  }
});

document.getElementById("company-docx-web").addEventListener("click", async () => {
  const company_name = document.getElementById("company-name").value.trim();
  if (!company_name) return setOutput("company-output", "Merci de renseigner un nom.");
  setOutput("company-output", "Génération du Word (web search)...");
  try {
    const res = await apiPost("/company-docx", { company_name, use_web_search: true });
    const blob = await res.blob();
    downloadBlob(blob, `${company_name}_fiche_societe.docx`.replace(/\s+/g, "_"));
    setOutput("company-output", "Fichier Word généré.");
  } catch (err) {
    setOutput("company-output", err.message);
  }
});

document.getElementById("funds-generate").addEventListener("click", async () => {
  const question = document.getElementById("funds-question").value.trim();
  if (!question) return setOutput("funds-output", "Merci de poser une question.");
  setOutput("funds-output", "Chargement...");
  try {
    const res = await apiPost("/funds", { question });
    const data = await res.json();
    setOutput("funds-output", data.answer);
  } catch (err) {
    setOutput("funds-output", err.message);
  }
});

document.getElementById("comparables-generate").addEventListener("click", async () => {
  const question = document.getElementById("comparables-question").value.trim();
  if (!question) return setOutput("comparables-output", "Merci de poser une question.");
  setOutput("comparables-output", "Chargement...");
  try {
    const res = await apiPost("/comparables", { question });
    const data = await res.json();
    setOutput("comparables-output", data.answer);
  } catch (err) {
    setOutput("comparables-output", err.message);
  }
});

document.getElementById("pdf-watermark").addEventListener("click", async () => {
  const file = document.getElementById("pdf-file").files[0];
  const bank_name = document.getElementById("bank-name").value.trim();
  if (!file || !bank_name) {
    return setOutput("pdf-output", "Veuillez choisir un PDF et un nom de banque.");
  }
  setOutput("pdf-output", "Traitement...");
  try {
    const pdf_base64 = await readFileAsBase64(file);
    const res = await apiPost("/watermark", { pdf_base64, bank_name });
    const data = await res.json();
    const pdfBytes = Uint8Array.from(atob(data.pdf_base64), (c) => c.charCodeAt(0));
    downloadBlob(new Blob([pdfBytes], { type: "application/pdf" }), `watermark_${file.name}`);
    setOutput("pdf-output", "PDF filigrané prêt.");
  } catch (err) {
    setOutput("pdf-output", err.message);
  }
});

document.getElementById("pdf-break").addEventListener("click", async () => {
  const file = document.getElementById("pdf-file").files[0];
  if (!file) return setOutput("pdf-output", "Veuillez choisir un PDF.");
  setOutput("pdf-output", "Traitement...");
  try {
    const pdf_base64 = await readFileAsBase64(file);
    const res = await apiPost("/break-pdf", { pdf_base64 });
    const data = await res.json();
    const pdfBytes = Uint8Array.from(atob(data.pdf_base64), (c) => c.charCodeAt(0));
    downloadBlob(new Blob([pdfBytes], { type: "application/pdf" }), `unlocked_${file.name}`);
    setOutput("pdf-output", "PDF prêt.");
  } catch (err) {
    setOutput("pdf-output", err.message);
  }
});

