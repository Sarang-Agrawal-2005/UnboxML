// Global variables
let currentDataset = null;
let currentModel = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initParticles();
    initPageNavigation();
    initTabs(); // Main pipeline tabs
    initThemeToggle();
    initMobileMenu();
    initSmoothScrolling();
    
    // Initialize preprocessing tabs with delay to avoid conflicts
    setTimeout(() => {
        initPreprocessingTabs();
        initPreprocessingControls();
        populateTargetDropdown();
    }, 200);
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

// Initialize tabs function - REPLACE EXISTING FUNCTION
function initTabs() {
    // Handle main pipeline tabs (dashboard tabs)
    const mainTabButtons = document.querySelectorAll('.dashboard-tabs .tab-btn');
    const mainTabContents = document.querySelectorAll('.dashboard-content .tab-content');
    
    // Initialize first main tab as active on page load
    if (mainTabButtons.length > 0 && mainTabContents.length > 0) {
        // Remove any existing active classes
        mainTabButtons.forEach(btn => btn.classList.remove('active'));
        mainTabContents.forEach(content => content.classList.remove('active'));
        
        // Set first tab as active
        mainTabButtons[0].classList.add('active');
        mainTabContents[0].classList.add('active');
    }
    
    // Add click event listeners for main tabs
    mainTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            // Remove active class from all main buttons and contents
            mainTabButtons.forEach(btn => btn.classList.remove('active'));
            mainTabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            const targetContent = document.getElementById(targetTab);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
    
    // Note: Preprocessing tabs are handled separately in initPreprocessingTabs()
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
const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const uploadSpinner  = document.getElementById('uploadSpinner');   // <i class="fas fa-spinner">
const uploadPreview  = document.getElementById('uploadPreview');

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // reset UI
  uploadPreview.innerHTML = '';
  uploadSpinner.style.display = 'flex';          // ⬅️ SHOW spinner

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('https://unboxml.onrender.com/upload', {
      method: 'POST',
      body  : formData
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    uploadSpinner.style.display = 'none';        // ⬅️ HIDE spinner

    /* ------- build dataset preview ------- */
     const previewHTML = `
      <div class="upload-success-badge" style = "justify-content : center">
        <i class="fas fa-check-circle" style = "justify-content : center"></i>
        Upload Successful
      </div>
      
      <div class="dataset-info-grid">
        <div class="info-stat-card">
          <span class="stat-value">${data.shape[0].toLocaleString()}</span>
          <div class="stat-label">Rows</div>
        </div>
        <div class="info-stat-card">
          <span class="stat-value">${data.shape[1]}</span>
          <div class="stat-label">Features</div>
        </div>
        <div class="info-stat-card">
          <span class="stat-value">${(data.size / 1024).toFixed(1)}KB</span>
          <div class="stat-label">File Size</div>
        </div>
      </div>
      
      <div class="dataset-preview-section">
        <div class="preview-header" style = "justify-content : center">
          <div class="logo" style="text-align: center;">Dataset <span class="highlight">Preview</span></div>
        </div>
    `;

    uploadPreview.innerHTML = previewHTML;

    // Enhanced table generation
    if (Array.isArray(data.preview) && data.preview.length) {
      const headers = Object.keys(data.preview[0]);
      const rows = data.preview.map(r =>
        `<tr>${headers.map(h => `<td>${r[h]}</td>`).join('')}</tr>`).join('');

      uploadPreview.insertAdjacentHTML('beforeend', `
        <div class="table-scroll">
          <table class="upload-preview-table">
            <thead>
              <tr>${headers.map(h => `<th><i class="fas fa-column"></i> ${h}</th>`).join('')}</tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        </div>
      `);
    } else {
      uploadPreview.insertAdjacentHTML('beforeend', '</div>');
    }

    /* ------- kick‑off downstream stages ------- */
    sessionStorage.setItem('uploadedFile', data.filename);
    analyzeData(data.filename);
    populateTargetDropdown();
    if (typeof refreshTrainingUI === 'function') refreshTrainingUI();

  } catch (err) {
    uploadSpinner.style.display = 'none';        // ⬅️ HIDE spinner on error
    uploadPreview.innerHTML = `<p style="color:red;">Failed to upload file: ${err.message}</p>`;
    console.error(err);
  }
});

document.getElementById('mockDatasetSelect').addEventListener('change', async function () {
  const selectedFile = this.value;
  if (!selectedFile) return;

  const uploadSpinner = document.getElementById('uploadSpinner');
  const uploadPreview = document.getElementById('uploadPreview');
  uploadPreview.innerHTML = '';
  uploadSpinner.style.display = 'flex';

  try {
    // Fetch the file from your mock_data directory
    const res = await fetch(`mock_data/${selectedFile}`);
    if (!res.ok) throw new Error(`Failed to fetch ${selectedFile}`);

    const blob = await res.blob();
    const file = new File([blob], selectedFile, { type: 'text/csv' });

    const formData = new FormData();
    formData.append('file', file);

    const uploadRes = await fetch('https://unboxml.onrender.com/upload', {
      method: 'POST',
      body: formData
    });

    if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status}`);
    const data = await uploadRes.json();

    uploadSpinner.style.display = 'none';

    // Enhanced preview generation in your upload handler
    const previewHTML = `
      <div class="upload-success-badge" style = "justify-content : center">
        <i class="fas fa-check-circle" style = "justify-content : center"></i>
        Upload Successful
      </div>
      
      <div class="dataset-info-grid">
        <div class="info-stat-card">
          <span class="stat-value">${data.shape[0].toLocaleString()}</span>
          <div class="stat-label">Rows</div>
        </div>
        <div class="info-stat-card">
          <span class="stat-value">${data.shape[1]}</span>
          <div class="stat-label">Features</div>
        </div>
        <div class="info-stat-card">
          <span class="stat-value">${(data.size / 1024).toFixed(1)}KB</span>
          <div class="stat-label">File Size</div>
        </div>
      </div>
      
      <div class="dataset-preview-section">
        <div class="preview-header" style = "justify-content : center">
          <div class="logo" style="text-align: center;">Dataset <span class="highlight">Preview</span></div>
        </div>
    `;

    uploadPreview.innerHTML = previewHTML;

    // Enhanced table generation
    if (Array.isArray(data.preview) && data.preview.length) {
      const headers = Object.keys(data.preview[0]);
      const rows = data.preview.map(r =>
        `<tr>${headers.map(h => `<td>${r[h]}</td>`).join('')}</tr>`).join('');

      uploadPreview.insertAdjacentHTML('beforeend', `
        <div class="table-scroll">
          <table class="upload-preview-table">
            <thead>
              <tr>${headers.map(h => `<th><i class="fas fa-column"></i> ${h}</th>`).join('')}</tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        </div>
      `);
    } else {
      uploadPreview.insertAdjacentHTML('beforeend', '</div>');
    }


    sessionStorage.setItem('uploadedFile', data.filename);
    analyzeData(data.filename);
    populateTargetDropdown();
    if (typeof refreshTrainingUI === 'function') refreshTrainingUI();

  } catch (err) {
    uploadSpinner.style.display = 'none';
    uploadPreview.innerHTML = `<p style="color:red;">${err.message}</p>`;
    console.error(err);
  }
});


/* drag & drop passthrough */
uploadZone.addEventListener('click',   () => fileInput.click());
uploadZone.addEventListener('dragover',e => { e.preventDefault(); uploadZone.classList.add('hovered'); });
uploadZone.addEventListener('dragleave',() => uploadZone.classList.remove('hovered'));
uploadZone.addEventListener('drop',    e => {
  e.preventDefault();
  uploadZone.classList.remove('hovered');
  if (e.dataTransfer.files[0]) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});


      

/* ----------  ANALYZE SECTION  ---------- */
async function analyzeData(filename) {
  if (!filename) { alert('⚠️ No dataset uploaded!'); return; }

  const spinner        = document.getElementById('analyzeSpinner');   // <i class="fas fa-spinner">
  const heatmapDiv     = document.getElementById('corrHeatmap');
  const summaryDiv     = document.getElementById('dataSummary');
  const columnAnalysis = document.getElementById('columnAnalysis');

  /* reset UI & show spinner */
  spinner.style.display = 'flex';     // ⬅️ SHOW spinner
  heatmapDiv.innerHTML  = '';
  summaryDiv.innerHTML  = '';
  columnAnalysis.innerHTML = '';

  try {
    const res = await fetch('https://unboxml.onrender.com/analyze', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ filename })
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const data = await res.json();

    spinner.style.display = 'none';   // ⬅️ HIDE spinner

    /* cache for later steps */
    sessionStorage.setItem('analysisInfo', JSON.stringify(data));

    /* build correlation heat‑map */
    buildCorrelationHeatmap(data.corr_matrix, data.corr_columns);
    const placeholder = document.getElementById('heatmapPlaceholder');
    if (placeholder) placeholder.remove();


    /* populate “Select Column” dropdown */
    const select = document.getElementById('columnSelect');
    select.innerHTML = '<option value="">Select a Feature for Analysis</option>';
    data.columns.forEach(c =>
      select.insertAdjacentHTML('beforeend', `<option value="${c}">${c}</option>`));

    select.onchange = () => {
      if (select.value)
        fetchColumnAnalysis(filename, select.value).catch(console.error);
      else
        columnAnalysis.innerHTML = '';
    };

    /* update downstream UI */
    const ptype = detectProblemType();
    document.getElementById('problemTypeLabel').textContent = capitalizeFirstLetter(ptype);

    fillModelDropdown(ptype);

  } catch (err) {
    spinner.style.display = 'none';   // ⬅️ HIDE spinner on error
    summaryDiv.innerHTML =
      `<p style="color:red;">Failed to analyze dataset: ${err.message}</p>`;
    console.error(err);
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
  const res  = await fetch("https://unboxml.onrender.com/column_analysis", {
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


/* ---------- ENHANCED PREPROCESSING SECTION ---------- */

// Initialize preprocessing functionality
document.addEventListener("DOMContentLoaded", () => {
    initPreprocessingTabs();
    initPreprocessingControls();
    populateTargetDropdown();
});

function initPreprocessingTabs() {
    const tabButtons = document.querySelectorAll('.preprocessing-tabs .tab-btn');
    const tabContents = document.querySelectorAll('.preprocessing-tabs .tab-content');

    // Ensure we have both buttons and contents
    if (tabButtons.length === 0 || tabContents.length === 0) {
        console.log('Preprocessing tabs not found');
        return;
    }

    // Clear any existing event listeners to prevent conflicts
    tabButtons.forEach(btn => {
        btn.replaceWith(btn.cloneNode(true));
    });

    // Get fresh references after cloning
    const freshTabButtons = document.querySelectorAll('.preprocessing-tabs .tab-btn');
    const freshTabContents = document.querySelectorAll('.preprocessing-tabs .tab-content');

    // Initialize first tab as active
    freshTabButtons.forEach((btn, idx) => {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
        btn.setAttribute('tabindex', '-1');
    });
    
    freshTabContents.forEach(content => {
        content.classList.remove('active');
        content.setAttribute('hidden', '');
        content.style.display = 'none';
        content.style.visibility = 'hidden';
    });

    // Activate first tab
    if (freshTabButtons[0] && freshTabContents[0]) {
        freshTabButtons[0].classList.add('active');
        freshTabButtons[0].setAttribute('aria-selected', 'true');
        freshTabButtons[0].removeAttribute('tabindex');
        
        freshTabContents[0].classList.add('active');
        freshTabContents[0].removeAttribute('hidden');
        freshTabContents[0].style.display = 'block';
        freshTabContents[0].style.visibility = 'visible';
    }

    freshTabButtons.forEach((button, index) => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            
            const targetTab = button.dataset.tab;
            
            // Remove active class from all buttons
            freshTabButtons.forEach((btn, idx) => {
                btn.classList.remove('active');
                btn.setAttribute('aria-selected', 'false');
                btn.setAttribute('tabindex', '-1');
            });
            
            // Hide all tab contents
            freshTabContents.forEach(content => {
                content.classList.remove('active');
                content.setAttribute('hidden', '');
                content.style.display = 'none';
                content.style.visibility = 'hidden';
            });
            
            // Activate clicked button and corresponding content
            button.classList.add('active');
            button.setAttribute('aria-selected', 'true');
            button.removeAttribute('tabindex');
            
            const targetContent = document.getElementById(targetTab);
            if (targetContent) {
                targetContent.classList.add('active');
                targetContent.removeAttribute('hidden');
                targetContent.style.display = 'block';
                targetContent.style.visibility = 'visible';
                
                // Force reflow to ensure display
                targetContent.offsetHeight;
            }
        });
    });
}



function initPreprocessingControls() {
    // Missing value method change handler
    const missingValueMethod = document.getElementById('missingValueMethod');
    const constantValueGroup = document.getElementById('constantValueGroup');
    const knnNeighborsGroup = document.getElementById('knnNeighborsGroup');
    
    if (missingValueMethod) {
        missingValueMethod.addEventListener('change', function() {
            constantValueGroup.style.display = this.value === 'constant' ? 'block' : 'none';
            knnNeighborsGroup.style.display = this.value === 'knn' ? 'block' : 'none';
        });
    }

    // Outlier method change handler
    const outlierMethod = document.getElementById('outlierMethod');
    const outlierThresholdGroup = document.getElementById('outlierThresholdGroup');
    const zscoreThresholdGroup = document.getElementById('zscoreThresholdGroup');
    
    if (outlierMethod) {
        outlierMethod.addEventListener('change', function() {
            const showIQR = this.value === 'iqr';
            const showZScore = this.value === 'zscore';
            
            outlierThresholdGroup.style.display = showIQR ? 'block' : 'none';
            zscoreThresholdGroup.style.display = showZScore ? 'block' : 'none';
        });
    }

    // Encoding method change handler
    const encodingMethod = document.getElementById('encodingMethod');
    const targetEncodingOptions = document.getElementById('targetEncodingOptions');
    
    if (encodingMethod) {
        encodingMethod.addEventListener('change', function() {
            targetEncodingOptions.style.display = this.value === 'target' ? 'block' : 'none';
        });
    }

    // Scaling method change handler
    const scalingMethod = document.getElementById('scalingMethod');
    const minmaxRangeGroup = document.getElementById('minmaxRangeGroup');
    const normalizerNormGroup = document.getElementById('normalizerNormGroup');
    
    if (scalingMethod) {
        scalingMethod.addEventListener('change', function() {
            minmaxRangeGroup.style.display = this.value === 'minmax' ? 'block' : 'none';
            normalizerNormGroup.style.display = this.value === 'normalizer' ? 'block' : 'none';
        });
    }

    // Initialize all sliders with value display
    initSliders();
    
    // Apply preprocessing button handler
    const applyBtn = document.getElementById("applyPreprocessing");
    if (applyBtn) {
        applyBtn.addEventListener("click", applyPreprocessing);
    }
}

function initSliders() {
    const sliders = [
        { id: 'knnNeighbors', valueId: 'knnNeighborsValue' },
        { id: 'outlierThreshold', valueId: 'outlierThresholdValue' },
        { id: 'zscoreThreshold', valueId: 'zscoreThresholdValue' },
        { id: 'targetEncodingSmoothing', valueId: 'targetEncodingSmoothingValue' },
        { id: 'multicollinearThreshold', valueId: 'multicollinearThresholdValue' },
        { id: 'lowCorrThreshold', valueId: 'lowCorrThresholdValue' },
        { id: 'varianceThreshold', valueId: 'varianceThresholdValue' },
        { id: 'kBestFeatures', valueId: 'kBestFeaturesValue' },
        { id: 'rfeFeatures', valueId: 'rfeFeaturesValue' },
        { id: 'rfeStep', valueId: 'rfeStepValue' }
    ];

    sliders.forEach(slider => {
        const sliderElement = document.getElementById(slider.id);
        const valueElement = document.getElementById(slider.valueId);
        
        if (sliderElement && valueElement) {
            sliderElement.addEventListener('input', function() {
                valueElement.textContent = this.value;
            });
        }
    });
}

async function applyPreprocessing() {
    const filename = sessionStorage.getItem("uploadedFile") || sessionStorage.getItem("processedFile");
    if (!filename) {
        showPreprocessError("⚠️ Please upload a dataset first.");
        return;
    }

    const target = document.getElementById("preprocessTarget").value;
    if (!target) {
        showPreprocessError("⚠️ Please select a target column.");
        return;
    }

    // Show spinner
    const spinner = document.getElementById("preprocessSpinner");
    const applyBtn = document.getElementById("applyPreprocessing");
    spinner.style.display = "flex";
    applyBtn.disabled = true;

    // Gather all preprocessing options
    const preprocessingOptions = gatherPreprocessingOptions();
    preprocessingOptions.filename = filename;
    preprocessingOptions.target = target;

    try {
        const response = await fetch("https://unboxml.onrender.com/preprocess_enhanced", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(preprocessingOptions)
        });

        if (!response.ok) {
            throw new Error(`Server error ${response.status}`);
        }

        const result = await response.json();
        
        if (result.error) {
            showPreprocessError(result.error);
            return;
        }

        // Show success results
        showPreprocessSuccess(result);
        
        // Store processed file info
        sessionStorage.setItem("processedFile", result.filename);
        sessionStorage.setItem("preprocessTarget", target);
        
        // Refresh training UI if function exists
        if (typeof refreshTrainingUI === 'function') {
            refreshTrainingUI();
        }

    } catch (error) {
        console.error("Preprocessing error:", error);
        showPreprocessError("Failed to preprocess dataset. Please check your settings and try again.");
    } finally {
        spinner.style.display = "none";
        applyBtn.disabled = false;
    }
}

function gatherPreprocessingOptions() {
    return {
        // Data Cleaning
        missing_value_method: document.getElementById("missingValueMethod")?.value || "drop",
        constant_value: document.getElementById("constantValue")?.value || "",
        knn_neighbors: parseInt(document.getElementById("knnNeighbors")?.value || "5"),
        duplicate_method: document.getElementById("duplicateMethod")?.value || "none",
        outlier_method: document.getElementById("outlierMethod")?.value || "none",
        outlier_threshold: parseFloat(document.getElementById("outlierThreshold")?.value || "1.5"),
        zscore_threshold: parseFloat(document.getElementById("zscoreThreshold")?.value || "3"),
        
        // Encoding
        encoding_method: document.getElementById("encodingMethod")?.value || "label",
        handle_unknown: document.getElementById("handleUnknown")?.checked || false,
        target_encoding_smoothing: parseFloat(document.getElementById("targetEncodingSmoothing")?.value || "1"),
        
        // Feature Selection
        remove_multicollinear: document.getElementById("removeMulticollinear")?.checked || false,
        multicollinear_threshold: parseFloat(document.getElementById("multicollinearThreshold")?.value || "0.95"),
        remove_low_corr: document.getElementById("removeLowCorr")?.checked || false,
        low_corr_threshold: parseFloat(document.getElementById("lowCorrThreshold")?.value || "0.1"),
        remove_constant: document.getElementById("removeConstant")?.checked || false,
        remove_low_variance: document.getElementById("removeLowVariance")?.checked || false,
        variance_threshold: parseFloat(document.getElementById("varianceThreshold")?.value || "0.01"),
        select_k_best: document.getElementById("selectKBest")?.checked || false,
        k_best_features: parseInt(document.getElementById("kBestFeatures")?.value || "10"),


        // Recursive Feature Elimination
        enable_rfe: document.getElementById("enableRFE")?.checked || false,
        rfe_features: parseInt(document.getElementById("rfeFeatures")?.value || "10"),
        rfe_estimator: document.getElementById("rfeEstimator")?.value || "linear",
        rfe_step: parseInt(document.getElementById("rfeStep")?.value || "1"),
        
        // Feature Scaling
        scaling_method: document.getElementById("scalingMethod")?.value || "none",
        minmax_min: parseFloat(document.getElementById("minmaxMin")?.value || "0"),
        minmax_max: parseFloat(document.getElementById("minmaxMax")?.value || "1"),
        normalizer_norm: document.getElementById("normalizerNorm")?.value || "l2"
    };
}

function showPreprocessError(message) {
    const resultDiv = document.getElementById("preprocessResult");
    resultDiv.innerHTML = `<div class="error-message">${message}</div>`;
    resultDiv.classList.add("show");
}

function showPreprocessSuccess(result) {
    const resultDiv = document.getElementById("preprocessResult");
    
    let appliedSteps = result.applied_steps || [];
    let stepsHtml = appliedSteps.length > 0 ? 
        `<h4>Applied Preprocessing Steps:</h4><ul>${appliedSteps.map(step => `<li>${step}</li>`).join('')}</ul>` : 
        '';
    
    resultDiv.innerHTML = `
        <div class="success-message">
            <h3>✅ Preprocessing Completed Successfully!</h3>
            <div class="preprocessing-summary">
                <p><strong>Original Shape:</strong> ${result.original_shape[0]} rows × ${result.original_shape[1]} columns</p>
                <p><strong>Final Shape:</strong> ${result.final_shape[0]} rows × ${result.final_shape[1]} columns</p>
                <p><strong>Rows Removed:</strong> ${result.original_shape[0] - result.final_shape[0]}</p>
                <p><strong>Features Removed:</strong> ${result.original_shape[1] - result.final_shape[1]}</p>
                ${stepsHtml}
                <p><strong>Remaining Features:</strong> ${result.remaining_features.join(", ")}</p>
            </div>
        </div>
    `;
    resultDiv.classList.add("show");
    
    // Show temporary success indicator
    const indicator = document.getElementById("preprocessIndicator");
    indicator.textContent = "✅ Preprocessing completed successfully!";
    indicator.style.color = "var(--success)";
    setTimeout(() => {
        indicator.textContent = "";
    }, 4000);
}

async function populateTargetDropdown() {
    const filename = sessionStorage.getItem("uploadedFile");
    if (!filename) return;

    try {
        const response = await fetch("https://unboxml.onrender.com/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename })
        });

        if (!response.ok) throw new Error(`Server error ${response.status}`);

        const data = await response.json();
        const select = document.getElementById("preprocessTarget");

        // Create placeholder option with proper attributes
        const placeholderOption = '<option value="" disabled selected hidden>Choose Target Feature</option>';

        // Build complete options list
        let optionsHTML = placeholderOption;
        data.columns.forEach(col => {
            optionsHTML += `<option value="${col}">${col}</option>`;
        });

        // Set all options at once
        select.innerHTML = optionsHTML;

        // Force placeholder to be selected
        select.selectedIndex = 0;




    } catch (error) {
        console.error("Error populating target dropdown:", error);
    }
}


/* ----------  DROPDOWN POPULATION  ---------- */
// async function populateTargetDropdown() {
//   const filename = sessionStorage.getItem("uploadedFile");
//   if (!filename) return;

//   try {
//     const res = await fetch("https://unboxml.onrender.com/analyze", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ filename })
//     });
//     if (!res.ok) throw new Error(`Server error ${res.status}`);
//     const data = await res.json();

//     const select = document.getElementById("preprocessTarget");
//     select.innerHTML = "";
//     data.columns.forEach(col => {
//       select.insertAdjacentHTML("beforeend",
//         `<option value="${col}">${col}</option>`);
//     });
//   } catch (err) {
//     console.error(err);
//   }
// }

// function debugPreprocessingTabs() {
//     const tabButtons = document.querySelectorAll('.preprocessing-tabs .tab-btn');
//     const tabContents = document.querySelectorAll('.preprocessing-tabs .tab-content');
    
//     console.log('Tab buttons found:', tabButtons.length);
//     console.log('Tab contents found:', tabContents.length);
    
//     tabContents.forEach((content, index) => {
//         console.log(`Tab ${index}:`, {
//             id: content.id,
//             display: getComputedStyle(content).display,
//             visibility: getComputedStyle(content).visibility,
//             hasActiveClass: content.classList.contains('active')
//         });
//     });
// }

// Call this function in browser console if issues persist

/* ----------  TRAIN MODEL SECTION  ---------- */

const MODEL_MAP = {
  regression : {
    "Linear Regression"      : {id:"LinearRegression",   params:{}},
    "Ridge Regression"       : {id:"Ridge",             params:{alpha:[0.1,1,0.1]}},
    "Lasso Regression"       : {id:"Lasso",             params:{alpha:[0.1,1,0.1]}},
    "ElasticNet"             : {id:"ElasticNet",        params:{alpha:[0.1,1,0.1], l1_ratio:[0,0.5,0.05]}},
    "Decision Tree"          : {id:"DecisionTreeReg",   params:{max_depth:[1,10,1]}},
    "Random Forest"          : {id:"RandomForestReg",   params:{n_estimators:[5,50,5], max_depth:[1,5,1]}},
    "Gradient Boosting"      : {id:"GradBoostReg",      params:{n_estimators:[5,50,5], learning_rate:[0.05,0.2,0.05]}},
    "K‑Nearest Neighbors"    : {id:"KNNReg",            params:{n_neighbors:[1,10,1]}},
    "Support Vector Machine" : {id:"SVMReg",            params:{C:[0.1,5,0.1]}},
    "XGBoost"                : {id:"XGBReg",            params:{n_estimators:[5,50,5], learning_rate:[0.05,0.2,0.05]}},
    "Neural Network"         : {id:"MLPReg",            params:{hidden_layer_sizes:[1,3,1]}},
  },
  classification : {
    "Logistic Regression"    : {id:"LogisticRegression", params:{C:[0.1,5,0.1]}},
    "Naive Bayes"            : {id:"NaiveBayes",        params:{}},
    "Decision Tree"          : {id:"DecisionTree",      params:{max_depth:[1,10,1]}},
    "Random Forest"          : {id:"RandomForest",      params:{n_estimators:[5,50,5], max_depth:[1,5,1]}},
    "Gradient Boosting"      : {id:"GradBoost",         params:{n_estimators:[5,50,5], learning_rate:[0.05,0.2,0.05]}},
    "K‑Nearest Neighbors"    : {id:"KNN",               params:{n_neighbors:[1,10,1]}},
    "Support Vector Machine" : {id:"SVM",               params:{C:[0.1,5,0.1]}},
    "XGBoost"                : {id:"XGB",               params:{n_estimators:[5,50,5], learning_rate:[0.05,0.2,0.05]}},
    "Neural Network"         : {id:"MLP",               params:{hidden_layer_sizes:[1,3,1]}},
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
  document.getElementById("problemTypeLabel").textContent = capitalizeFirstLetter(ptype);
  //fillModelDropdown(ptype);
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

    const response = await fetch("https://unboxml.onrender.com/train",{
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
  document.getElementById("problemTypeLabel").textContent = capitalizeFirstLetter(ptype);

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
    const response = await fetch("https://unboxml.onrender.com/evaluate", {
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
    document.getElementById("evaluationSummary").innerHTML = `<p>${capitalizeFirstLetter(problemType)} Metrics</p>`;


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
  link.href = `https://unboxml.onrender.com/download_model?model_id=${modelID}`;
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
      const res  = await fetch("https://unboxml.onrender.com/generate_streamlit", {
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
      const res = await fetch("https://unboxml.onrender.com/generate_api_script", {
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



// Add this function to handle mobile chart responsiveness
function makeChartsResponsive() {
  // Configure responsive layout for all Plotly charts
  const responsiveConfig = {
    responsive: true,
    displayModeBar: false,
    staticPlot: false
  };
  
  const mobileLayout = {
    autosize: true,
    margin: { l: 40, r: 20, t: 40, b: 40 },
    font: { size: 10 }
  };
  
  // Apply to existing chart functions
  window.addEventListener('resize', function() {
    if (window.innerWidth <= 768) {
      // Relayout existing charts for mobile
      const charts = document.querySelectorAll('.js-plotly-plot');
      charts.forEach(chart => {
        if (chart.layout) {
          Plotly.relayout(chart, mobileLayout);
        }
      });
    }
  });
}

// Update your existing chart functions
function buildCorrelationHeatmap(matrix, labels) {
  const data = [{
    z: matrix,
    x: labels,
    y: labels,
    type: "heatmap",
    colorscale: [
      [0, "rgba(0,200,255,0.2)"],
      [1, "rgba(157,78,221,0.9)"]
    ],
    hoverongaps: false
  }];
  
  const layout = {
    margin: window.innerWidth <= 768 ? 
      { l: 60, r: 10, t: 10, b: 60 } : 
      { l: 90, r: 20, t: 10, b: 90 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { 
      color: getCss("--text-primary", "#FFFFFF"),
      size: window.innerWidth <= 768 ? 8 : 12
    },
    xaxis: { 
      tickangle: -45,
      tickfont: { size: window.innerWidth <= 768 ? 8 : 10 }
    },
    yaxis: {
      tickfont: { size: window.innerWidth <= 768 ? 8 : 10 }
    },
    autosize: true
  };
  
  const config = {
    responsive: true,
    displayModeBar: false
  };
  
  Plotly.newPlot("corrHeatmap", data, layout, config);
}

// Call this function when the page loads
document.addEventListener('DOMContentLoaded', function() {
  makeChartsResponsive();
  // ... your existing initialization code
});

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}
