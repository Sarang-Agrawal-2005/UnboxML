// Global variables
let currentDataset = null;
let currentModel = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initParticles();
    initPageNavigation();
    initTabs();
    initThemeToggle();
    initMobileMenu();
    initSmoothScrolling();
});

// Initialize particles background
function initParticles() {
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: '#00c8ff' },
                shape: { type: 'circle' },
                opacity: { value: 0.5, random: false },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#00c8ff',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 6,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'repulse' },
                    onclick: { enable: true, mode: 'push' },
                    resize: true
                }
            },
            retina_detect: true
        });
    }
}

// Initialize page navigation
function initPageNavigation() {
    // Get navigation buttons
    const startBuildingBtn = document.getElementById('startBuildingBtn');
    const getStartedBtn = document.getElementById('getStartedBtn');
    const ctaGetStartedBtn = document.getElementById('ctaGetStartedBtn');

    // Add event listeners to all navigation buttons
    const navButtons = [startBuildingBtn, getStartedBtn, ctaGetStartedBtn];
    navButtons.forEach((btn) => {
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                // showPlatform();
            });
        }
    });
}




// Initialize theme toggle
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;
    
    // Check for saved theme preference or default to dark
    const savedTheme = localStorage.getItem('theme') || 'dark';
    body.setAttribute('data-theme', savedTheme);
    
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Add animation class
            body.classList.add('theme-transition');
            setTimeout(() => {
                body.classList.remove('theme-transition');
            }, 300);
        });
    }
}

// Initialize mobile menu
function initMobileMenu() {
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            mobileMenuBtn.classList.toggle('active');
        });
        
        // Close mobile menu when clicking on a link
        const navItems = navLinks.querySelectorAll('a');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                navLinks.classList.remove('active');
                mobileMenuBtn.classList.remove('active');
            });
        });
    }
}

// Initialize smooth scrolling
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href === '#' || href === '#platform') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialize tabs
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

const getStartedBtn = document.getElementById("getStartedBtn");
if (getStartedBtn) {
  getStartedBtn.onclick = () => {
    window.location.href = "pipeline.html";
  };
}

const startBuildingBtn = document.getElementById("startBuildingBtn");
if (startBuildingBtn) {
  startBuildingBtn.onclick = () => {
    window.location.href = "pipeline.html";
  };
}

const ctaGetStartedBtn = document.getElementById("ctaGetStartedBtn");
if (ctaGetStartedBtn) {
  ctaGetStartedBtn.onclick = () => {
    window.location.href = "pipeline.html";
  };
}


/* ----------  UPLOAD SECTION  ---------- */
document.getElementById('fileInput').addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  fetch('http://127.0.0.1:5000/upload', {
    method: 'POST',
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      const previewHTML = `
        <p><strong><span class="highlight">File Uploaded - </span></strong> ${data.filename}</p>
        
        <p><strong><span class="highlight">Features - </span></strong> ${data.shape[1]}</p>
        <p><strong><span class="highlight">Rows - </span></strong> ${data.shape[0]}</p>
        <p><strong><span class="highlight">Dataset Sample:</span></strong></p>
       
      `;
      document.getElementById('uploadPreview').innerHTML = previewHTML;
      if (data.preview && data.preview.length > 0) {
        const headers = Object.keys(data.preview[0]);
        const rows = data.preview.map(row => {
          return `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`;
        }).join('');

        const table = `
        <div class="table-scroll">
          <table class="upload-preview-table">
            <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
      uploadPreview.innerHTML += table;

      }
      analyzeData(data.filename); // from upload response
      sessionStorage.setItem("uploadedFile", data.filename);
      populateTargetDropdown();
      refreshTrainingUI(); 
      })
    .catch(err => {
      document.getElementById('uploadPreview').innerHTML = '<p style="color:red;">Failed to upload file</p>';
      console.error(err);
    });
});



const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadSpinner = document.getElementById('uploadSpinner');
const uploadPreview = document.getElementById('uploadPreview');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('hovered');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('hovered');
});
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('hovered');
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

      

/* ----------  ANALYZE SECTION  ---------- */
async function analyzeData(filename) {
  if (!filename) {
    alert("⚠️ No dataset uploaded!");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename })
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const data = await response.json();

    // ✅ Store for use in preprocessing, training, evaluation
    sessionStorage.setItem("analysisInfo", JSON.stringify(data));

     buildCorrelationHeatmap(data.corr_matrix, data.corr_columns);

    const select = document.getElementById("columnSelect");
    select.innerHTML = '<option value="">-- choose column --</option>';
    data.columns.forEach(c =>
      select.insertAdjacentHTML("beforeend", `<option value="${c}">${c}</option>`));

    select.onchange = () => {
      if (select.value) fetchColumnAnalysis(filename, select.value)
        .catch(err => console.error("Column analysis error:", err));
      else document.getElementById("columnAnalysis").innerHTML = "";
    };

    let html = `
      
    `;

    document.getElementById("dataSummary").innerHTML = html;

    // ✅ Optional: refresh UI if needed
    const ptype = detectProblemType();
    document.getElementById("problemTypeLabel").textContent = ptype;
    fillModelDropdown(ptype);

  } catch (error) {
    console.error("❌ Analyze error:", error);
    document.getElementById("dataSummary").innerHTML =
      `<p style="color:red;">Failed to analyze dataset: ${error.message}</p>`;
  }
}

/* ────────────  PLOT HELPERS  ─────────────────────────────────────────────── */

function getCss(varName, fallback="#fff") {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim() || fallback;
}

/* === Heat‑map === */
function buildCorrelationHeatmap(matrix, labels) {
  const data = [{
    z         : matrix,
    x         : labels,
    y         : labels,
    type      : "heatmap",
    colorscale: [
      [0,  "rgba(0,200,255,0.2)"],
      [1,  "rgba(157,78,221,0.9)"]
    ],
    hoverongaps:false
  }];

  const layout = {
    margin        : {l:90,r:20,t:10,b:90},
    paper_bgcolor : "rgba(0,0,0,0)",
    plot_bgcolor  : "rgba(0,0,0,0)",
    font          : {color: getCss("--text-primary", "#FFFFFF")},
    xaxis         : {tickangle:-45}
  };

  Plotly.newPlot("corrHeatmap", data, layout, {responsive:true});
}

/* === Histogram === */
function buildHistogram(bins, counts, col) {
  const data = [{x: bins.slice(0,-1),  // last edge unused
                 y: counts,
                 type:"bar",
                 marker:{line:{width:0}}}];

  const layout = {
    title         : `Distribution of ${col}`,
    paper_bgcolor : "rgba(0,0,0,0)",
    plot_bgcolor  : "rgba(0,0,0,0)",
    font          : {color:getCss("--text-primary", "#fff")},
    margin        : {t:40,l:50,r:20,b:50}
  };
  Plotly.newPlot("columnChart", data, layout, {responsive:true});
}

/* === Pie === */
function buildPie(labels, values, col) {
  const data = [{
    labels : labels,
    values : values,
    type   : "pie",
    hole   : 0.35,
    textinfo:"label+percent"
  }];
  const layout = {
    title         : `${col} – category share`,
    paper_bgcolor : "rgba(0,0,0,0)",
    font          : {color:getCss("--text-primary","#fff")},
    margin        : {t:40,l:20,r:20,b:20}
  };
  Plotly.newPlot("columnChart", data, layout, {responsive:true});
}

function plotLayout(title, xTitle="", yTitle="") {
  return {
    title,
    paper_bgcolor : "rgba(0,0,0,0)",
    plot_bgcolor  : "rgba(0,0,0,0)",
    font          : {color: getCss("--text-primary", "#fff")},
    xaxis         : {title: xTitle, tickfont:{color:getCss("--text-primary")}},
    yaxis         : {title: yTitle, tickfont:{color:getCss("--text-primary")}},
    margin        : {t:40,l:50,r:20,b:50},
  };
}


/* === Column‑analysis orchestration === */
async function fetchColumnAnalysis(filename, column) {
  const res  = await fetch("http://127.0.0.1:5000/column_analysis", {
                 method : "POST",
                 headers: {"Content-Type":"application/json"},
                 body   : JSON.stringify({filename, column})
               });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  // Prepare container
  const el   = document.getElementById("columnAnalysis");
  el.innerHTML = `<div id="columnChart" style="height:400px;"></div>
                  <pre id="columnStats" style="white-space:pre-wrap;"></pre>`;

  // Draw chart
  if (data.dtype === "numeric") {
    buildHistogram(data.hist.bins, data.hist.counts, column);
  } else {
    const labels = Object.keys(data.counts);
    const values = Object.values(data.counts);
    buildPie(labels, values, column);
  }
  // Stats text
  document.getElementById("columnStats").textContent =
      JSON.stringify(data.stats, null, 2);
}


/* ----------  PREPROCESSING SECTION  ---------- */

/* helper: show error in the same box every time */
function showPreprocessError(msg) {
  document.getElementById("preprocessResult").innerHTML =
    `<p style="color:red;">${msg}</p>`;
}

/*  runs after the DOM is fully parsed  */
document.addEventListener("DOMContentLoaded", () => {

  /* (re‑)populate the dropdown whenever the page loads */
  populateTargetDropdown();

  /* safe attachment – only after DOM is ready */
  const applyBtn = document.getElementById("applyPreprocessing");
  if (!applyBtn) return;   // guard in case markup changes

  applyBtn.addEventListener("click", async () => {
    /* quick validations */
    const filename = sessionStorage.getItem("uploadedFile");
    if (!filename) { showPreprocessError("⚠️ Please upload a dataset first."); return; }

    const target = document.getElementById("preprocessTarget").value;
    sessionStorage.setItem("preprocessTarget", target);  // ✅ Set the selected target
    refreshTrainingUI(); 
    if (!target) { showPreprocessError("⚠️ Choose a target column."); return; }

    /* gather options */
    const payload = {
      filename,
      target,
      dropNA:           document.getElementById("dropNA").checked,
      removeDuplicates: document.getElementById("removeDuplicates").checked,
      removeOutliers:   document.getElementById("removeOutliers").checked,
      dropLowCorr:      document.getElementById("dropLowCorr").checked,
      normalize:        document.getElementById("normalize").checked,
      standardize:      document.getElementById("standardize").checked
    };

    try {
      const res = await fetch("http://127.0.0.1:5000/preprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const result = await res.json();

      if (result.error) { showPreprocessError(result.error); return; }

      /* success UI */
      document.getElementById("preprocessResult").innerHTML = `
        <p><strong><span class="highlight">Preprocessing completed!</span></strong></p>
        <p><strong><span class="highlight">Remaining Rows:</span></strong> ${result.shape[0]}</p>
        <p><strong><span class="highlight">Remaining Columns:</span></strong> ${result.shape[1]}</p>
        <p><strong><span class="highlight">Remaining Features:</span></strong> ${result.columns.join(", ")}</p>
      `;
      document.getElementById("preprocessIndicator").textContent =
        "✅ Preprocessing completed successfully!";
      setTimeout(() =>
        document.getElementById("preprocessIndicator").textContent = "", 4000);

      /* remember the new cleaned‑up file */
      sessionStorage.setItem("processedFile", result.filename);

    } catch (err) {
      console.error(err);
      showPreprocessError("Failed to preprocess dataset. Check console logs.");
    }
  });
});


/* ----------  DROPDOWN POPULATION  ---------- */
async function populateTargetDropdown() {
  const filename = sessionStorage.getItem("uploadedFile");
  if (!filename) return;

  try {
    const res = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename })
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    const select = document.getElementById("preprocessTarget");
    select.innerHTML = "";
    data.columns.forEach(col => {
      select.insertAdjacentHTML("beforeend",
        `<option value="${col}">${col}</option>`);
    });
  } catch (err) {
    console.error(err);
  }
}


/* ----------  TRAIN MODEL SECTION  ---------- */

const MODEL_MAP = {
  regression : {
    "Linear Regression"      : {id:"LinearRegression",   params:{ }},
    "Ridge Regression"       : {id:"Ridge",             params:{alpha:[0.1,10,0.1]}},
    "Lasso Regression"       : {id:"Lasso",             params:{alpha:[0.1,10,0.1]}},
    "ElasticNet"             : {id:"ElasticNet",        params:{alpha:[0.1,10,0.1], l1_ratio:[0,1,0.01]}},
    "Decision Tree"          : {id:"DecisionTreeReg",   params:{max_depth:[1,30,1]}},
    "Random Forest"          : {id:"RandomForestReg",   params:{n_estimators:[10,300,10], max_depth:[1,30,1]}},
    "Gradient Boosting"      : {id:"GradBoostReg",      params:{n_estimators:[50,400,25], learning_rate:[0.01,0.3,0.01]}},
    "K‑Nearest Neighbors"    : {id:"KNNReg",            params:{n_neighbors:[1,30,1]}},
    "Support Vector Machine" : {id:"SVMReg",            params:{C:[0.1,10,0.1]}},
    "XGBoost"                : {id:"XGBReg",            params:{n_estimators:[50,400,25], learning_rate:[0.01,0.3,0.01]}},
    "Neural Network"         : {id:"MLPReg",            params:{hidden_layer_sizes:[1,20,1]}},
  },
  classification : {
    "Logistic Regression"    : {id:"LogisticRegression", params:{C:[0.1,10,0.1]}},
    "Naive Bayes"            : {id:"NaiveBayes",        params:{ } },
    "Decision Tree"          : {id:"DecisionTree",      params:{max_depth:[1,30,1]}},
    "Random Forest"          : {id:"RandomForest",      params:{n_estimators:[10,300,10], max_depth:[1,30,1]}},
    "Gradient Boosting"      : {id:"GradBoost",         params:{n_estimators:[50,400,25], learning_rate:[0.01,0.3,0.01]}},
    "K‑Nearest Neighbors"    : {id:"KNN",               params:{n_neighbors:[1,30,1]}},
    "Support Vector Machine" : {id:"SVM",               params:{C:[0.1,10,0.1]}},
    "XGBoost"                : {id:"XGB",               params:{n_estimators:[50,400,25], learning_rate:[0.01,0.3,0.01]}},
    "Neural Network"         : {id:"MLP",               params:{hidden_layer_sizes:[1,20,1]}},
  }
};

/* auto‑detect problem type based on target dtype saved earlier */
function detectProblemType() {
  const info = JSON.parse(sessionStorage.getItem("analysisInfo")||"{}");
  const target = sessionStorage.getItem("preprocessTarget") || sessionStorage.getItem("uploadedTarget");
  if (!target || !info.dtypes) return "regression";

  const dtype = info.dtypes[target];    // e.g. 'object', 'int64', 'float64'
  if (dtype === "object" || dtype === "bool") return "classification";
  // heuristic: small number of unique values → classification
  const uniques = info.unique_counts ? info.unique_counts[target] : null;
  if (uniques && uniques < 20) return "classification";
  return "regression";
}

/* populate models for detected problem type */
function fillModelDropdown(ptype) {
  const select = document.getElementById("modelSelect");
  select.innerHTML = "";
  Object.keys(MODEL_MAP[ptype]).forEach(name=>{
    select.insertAdjacentHTML("beforeend", `<option value="${name}">${name}</option>`);
  });
  buildHyperSliders(); // first model
}

/* build hyper‑parameter sliders dynamically */
function buildHyperSliders() {
  const select = document.getElementById("modelSelect");
  const modelName = select.value;
  const ptype = detectProblemType();
  const modelMeta = MODEL_MAP[ptype][modelName];
  const hpDiv = document.getElementById("hyperparamControls");
  hpDiv.innerHTML = "";

  Object.entries(modelMeta.params).forEach(([hparam, [min,max,step]])=>{
    const id = `hp_${hparam}`;
    hpDiv.insertAdjacentHTML("beforeend",`
      <div class="hyper-slider">
        <label for="${id}">${hparam} (<span id="${id}_val">${min}</span>)</label>
        <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${min}">
      </div>
    `);
  });

  // live update label
  hpDiv.querySelectorAll("input[type=range]").forEach(r=>{
    r.addEventListener("input",()=> {
      document.getElementById(`${r.id}_val`).textContent = r.value;
    });
  });
}

document.addEventListener("DOMContentLoaded", ()=>{

  /* update on tab load / dataset change */
  const ptype = detectProblemType();
  document.getElementById("problemTypeLabel").textContent = ptype;
  fillModelDropdown(ptype);
  document.getElementById("modelSelect").addEventListener("change", buildHyperSliders);

  /* TRAIN BUTTON */
  
  document.getElementById("trainModel").addEventListener("click", async ()=>{
    if (!sessionStorage.getItem("processedFile")) {
      alert("⚠️ Please apply preprocessing before training.");
      return;
    }

    const ptype = detectProblemType();
    const modelName = document.getElementById("modelSelect").value;
    const modelMeta = MODEL_MAP[ptype][modelName];

    // gather hyper‑param values
    const params = {};
    Object.keys(modelMeta.params).forEach(hp=>{
      const val = document.getElementById(`hp_${hp}`).value;
      params[hp] = isNaN(val)?val:Number(val);
    });

    // UI spinner
    document.getElementById("trainSpinner").style.display = "block";
    document.getElementById("trainingOutput").textContent = "";

    const response = await fetch("http://127.0.0.1:5000/train",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        filename: sessionStorage.getItem("processedFile") || sessionStorage.getItem("uploadedFile"),
        target:   sessionStorage.getItem("preprocessTarget"),
        model_id: modelMeta.id,
        params
      })
    });
    const res = await response.json();
    document.getElementById("trainSpinner").style.display = "none";

    if (res.error){
      document.getElementById("trainingOutput").innerHTML = `<p style="color:red;">${res.error}</p>`;
    } else {
      document.getElementById("trainingOutput").innerHTML = `
        <p><strong>✅ ${modelName} trained successfully!</strong></p>
        

      `;
    }
    sessionStorage.setItem("trainedModelId", modelMeta.id);
    sessionStorage.setItem("trainedModelName", modelName); 
    sessionStorage.setItem("trainedModelParams", JSON.stringify(params));
  });
});

function refreshTrainingUI() {
  const ptype = detectProblemType();
  document.getElementById("problemTypeLabel").textContent = ptype;
  fillModelDropdown(ptype); // also builds the first model's sliders
}


/* ----------  EVALUATE SECTION  ---------- */

function renderMetrics(metrics) {
  // Simple formatter – could be improved into a table
  let html = "<ul>";
  Object.entries(metrics).forEach(([k,v])=>{
    if (k === "confusion_matrix") {
      html += `<li><strong>${k}</strong>:<br>${JSON.stringify(v)}</li>`;
    } else {
      html += `<li><strong>${k}</strong>: ${v.toFixed ? v.toFixed(4) : v}</li>`;
    }
  });
  html += "</ul>";
  return html;
}

document.getElementById("runEvaluation").addEventListener("click", async () => {
  if (!sessionStorage.getItem("processedFile")) {
    alert("⚠️ Train a model first.");
    return;
  }

  document.getElementById("evalSpinner").style.display = "block";
  document.getElementById("evaluationMetrics").innerHTML = "";
  document.getElementById("evaluationSummary").innerHTML = "";
  document.getElementById("evalCharts").innerHTML = "";

  const problemType = detectProblemType();
  const modelName = document.getElementById("modelSelect").value;
  const modelID = MODEL_MAP[problemType][modelName].id;

  try {
    const response = await fetch("http://127.0.0.1:5000/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: sessionStorage.getItem("processedFile"),
        target: sessionStorage.getItem("preprocessTarget"),
        model_id: modelID,
        problem_type: problemType
      })
    });

    const res = await response.json();
    document.getElementById("evalSpinner").style.display = "none";

    if (res.error) {
      document.getElementById("evaluationMetrics").innerHTML = `<p style="color:red;">${res.error}</p>`;
      return;
    }

    // Show problem type
    document.getElementById("evaluationSummary").innerHTML =
      `<p>${problemType.charAt(0).toUpperCase() + problemType.slice(1)} Metrics</p>`;

    // Show metrics table
    const metricsTable = [`<table class="eval-metrics-table">`];
    metricsTable.push();
    Object.entries(res.metrics).forEach(([k, v]) => {
      if (Array.isArray(v)) return; // skip confusion matrix here
      metricsTable.push(`<tr><td>${k}</td><td>${v.toFixed ? v.toFixed(4) : v}</td></tr>`);
    });
    metricsTable.push("</table>");
    document.getElementById("evaluationMetrics").innerHTML = metricsTable.join("");

    // Overfitting warning (if available)
    if (res.train_score && res.test_score && res.train_score - res.test_score > 0.1) {
      document.getElementById("evaluationSummary").innerHTML +=
        `<p style="color: orange;"><strong>⚠️ Possible Overfitting</strong>: Train score = ${res.train_score.toFixed(2)}, Test score = ${res.test_score.toFixed(2)}</p>`;
    }

    // Charts
    const plots = res.plots;
    const evalCharts = document.getElementById("evalCharts");
    const getCss = (v, fallback = "#fff") => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || fallback;
    const layout = (title, x, y) => ({
      title,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: getCss("--text-primary", "#fff") },
      xaxis: { title: x },
      yaxis: { title: y },
      margin: { t: 40, l: 50, r: 20, b: 50 }
    });

    if (problemType === "regression") {
      const sDiv = document.createElement("div");
      sDiv.id = "scatterPlot";
      sDiv.style.height = "400px";
      evalCharts.appendChild(sDiv);

      Plotly.newPlot("scatterPlot", [{
        x: plots.actual,
        y: plots.predicted,
        mode: "markers",
        marker: { color: getCss("--primary", "#00c8ff") }
      }], layout("Actual vs Predicted", "Actual", "Predicted"));

      const rDiv = document.createElement("div");
      rDiv.id = "residualPlot";
      rDiv.style.height = "400px";
      evalCharts.appendChild(rDiv);

      Plotly.newPlot("residualPlot", [{
        x: plots.predicted,
        y: plots.residuals,
        mode: "markers",
        marker: { color: getCss("--accent", "#ff6b6b") }
      }], layout("Residuals", "Predicted", "Residual"));
    }

    if (problemType === "classification" && plots.confusion) {
      const cDiv = document.createElement("div");
      cDiv.id = "confMatrix";
      cDiv.style.height = "400px";
      evalCharts.appendChild(cDiv);

      Plotly.newPlot("confMatrix", [{
        z: plots.confusion,
        type: "heatmap",
        colorscale: "Viridis"
      }], layout("Confusion Matrix", "", ""));
    }

  } catch (err) {
    console.error("❌ Evaluation failed:", err);
    document.getElementById("evalSpinner").style.display = "none";
    document.getElementById("evaluationMetrics").innerHTML = `<p style="color:red;">Backend error</p>`;
  }
});


/* ----------  DOWNLOAD MODEL SECTION  ---------- */

document.getElementById("downloadModel").addEventListener("click", () => {
  const problemType = detectProblemType(); // auto-detect based on target
  const modelName = document.getElementById("modelSelect").value;
  const modelID = MODEL_MAP[problemType][modelName].id;

  const link = document.createElement("a");
  link.href = `http://127.0.0.1:5000/download_model?model_id=${modelID}`;
  link.download = `${modelID}.pkl`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// ── Streamlit app download ───────────────────────────────────────────────────
const dlBtn = document.getElementById("downloadStreamlitBtn");
if (dlBtn) {
  dlBtn.addEventListener("click", async () => {
    const modelId = sessionStorage.getItem("trainedModelId");
    if (!modelId) { alert("⚠️ Train a model first."); return; }

    dlBtn.disabled = true;
    document.getElementById("downloadStatus").textContent = "⏳ Building package…";

    try {
      const res  = await fetch("http://127.0.0.1:5000/generate_streamlit", {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ model_id: modelId })
      });

      if (!res.ok) throw new Error((await res.json()).error);

      const blob = await res.blob();
      const url  = window.URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url;
      a.download = `${modelId}_streamlit_app.zip`;
      a.click();
      window.URL.revokeObjectURL(url);

      document.getElementById("downloadStatus")
              .textContent = "✅ Download ready!";
    } catch(err) {
      document.getElementById("downloadStatus")
              .textContent = "❌ " + err.message;
    } finally {
      dlBtn.disabled = false;
    }
  });
}

// ── API Script download ──────────────────────────────────────────────────────
const apiBtn = document.getElementById("downloadApiBtn");
if (apiBtn) {
  apiBtn.addEventListener("click", async () => {
    const modelId = sessionStorage.getItem("trainedModelId");
    if (!modelId) { alert("⚠️ Train a model first."); return; }

    apiBtn.disabled = true;
    document.getElementById("apiDownloadStatus").textContent = "⏳ Building API package…";

    try {
      const res = await fetch("http://127.0.0.1:5000/generate_api_script", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId })
      });

      if (!res.ok) throw new Error((await res.json()).error);

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${modelId}_api_script.zip`;
      a.click();
      window.URL.revokeObjectURL(url);

      document.getElementById("apiDownloadStatus").textContent = "✅ Download ready!";
    } catch (err) {
      document.getElementById("apiDownloadStatus").textContent = "❌ " + err.message;
    } finally {
      apiBtn.disabled = false;
    }
  });
}
