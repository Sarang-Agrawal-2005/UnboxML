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
        <p><strong>File:</strong> ${data.filename}</p>
        <p><strong>Columns:</strong> ${data.shape[1]}</p>
        <p><strong>Rows:</strong> ${data.shape[0]}</p>
        <p><strong>Features:</strong> ${data.columns.join(', ')}</p>
       
      `;
      document.getElementById('uploadPreview').innerHTML = previewHTML;
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

/*<h4>Preview:</h4>
        <table>
          <thead><tr>${Object.keys(data.preview[0]).map(k => `<th>${k}</th>`).join('')}</tr></thead>
          <tbody>
            ${data.preview.map(row => `<tr>${Object.values(row).map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table> */

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

    let html = `
      <h4>Numeric Features</h4>
      <pre>${JSON.stringify(data.describe_numeric, null, 2)}</pre>

      <h4>Categorical Features</h4>
      <pre>${JSON.stringify(data.describe_categorical, null, 2)}</pre>
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
        <p><strong>Preprocessing completed!</strong></p>
        <p><strong>Remaining Rows:</strong> ${result.shape[0]}</p>
        <p><strong>Remaining Columns:</strong> ${result.shape[1]}</p>
        <p><strong>Remaining Features:</strong> ${result.columns.join(", ")}</p>
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
  // guard: ensure model has been trained
  if (!sessionStorage.getItem("processedFile")) {
    alert("⚠️ Train a model first.");
    return;
  }

  document.getElementById("evalSpinner").style.display = "block";
  document.getElementById("evaluationMetrics").innerHTML = "";

  // detect problem type (regression/classification)
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
    } else {
      document.getElementById("evaluationMetrics").innerHTML = renderMetrics(res.metrics);
    }

  } catch (err) {
    console.error("❌ Evaluation failed:", err);
    document.getElementById("evalSpinner").style.display = "none";
    document.getElementById("evaluationMetrics").innerHTML = `<p style="color:red;">Failed to connect to backend</p>`;
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
