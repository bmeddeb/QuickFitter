// DOM Elements
const fileUpload = document.getElementById('file-upload');
const runButton = document.getElementById('run-button');
const resetButton = document.getElementById('reset-button');
const downloadButton = document.getElementById('download-button');
const modelSelect = document.getElementById('model-select');
const modelConfig = document.getElementById('model-config');
const nTermsSelect = document.getElementById('n-terms');
const fitMethodSelect = document.getElementById('fit-method');
const statusDiv = document.getElementById('status');
const statusMessage = document.getElementById('status-message');
const outputDiv = document.getElementById('output');
const plotOutput = document.getElementById('plot-output');
const reportOutput = document.getElementById('report-output');
const errorBox = document.getElementById('error-box');
const errorMessage = document.getElementById('error-message');

// Dynamic Parameter Controls
let paramControls = {};
let currentModel = 'djordjevic_sarkar';
let currentModelData = null;

// Global State
let pyodide = null;
let fileContent = null;
let isPyodideReady = false;
let currentData = null;
let originalFilename = '';

// Python modules to load (including __init__.py if you treat it as a package)
const pythonModules = [
    '__init__.py',
    'data_structures.py',
    'models.py',
    'optimization.py',
    'evaluation.py',
    'plotting.py',
    'main.py'
];

// Instead of executing each file immediately, write them into Pyodide's FS:
async function writeModuleToFS(moduleName) {
    const resp = await fetch(`python/${moduleName}`);
    if (!resp.ok) throw new Error(`Failed to fetch ${moduleName}: ${resp.statusText}`);
    const text = await resp.text();
    // ensure the /python directory exists
    try { pyodide.FS.mkdir('/python'); } catch(e) { /* already there */ }
    pyodide.FS.writeFile(`/python/${moduleName}`, text);
    console.log(`Wrote /python/${moduleName} to FS`);
}

async function main() {
    setLoadingState(true, 'Initializing Environment...');
    try {
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
        });

        setLoadingState(true, 'Loading Python Packages...');
        await pyodide.loadPackage(['numpy', 'pandas', 'micropip', 'matplotlib']);

        const micropip = pyodide.pyimport('micropip');
        setLoadingState(true, 'Installing lmfit & plotly...');
        await micropip.install(['lmfit', 'plotly']);

        setLoadingState(true, 'Writing Python modules into the virtual FS...');
        for (const module of pythonModules) {
            await writeModuleToFS(module);
        }

        // Now tell Python where to find them:
        await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/python")
        `);

        // Finally, import your entry-point in one go:
        await pyodide.runPythonAsync(`
from main import (
    load_data, run_analysis, calculate_model_from_params,
    create_updated_report, evaluate_current_fit,
    create_downloadable_plot, get_measured_data,
    get_available_models, run_analysis_with_model,
    calculate_model_from_params_generic
)
        `);

        isPyodideReady = true;
        setLoadingState(false);
        console.log('Pyodide environment ready!');

    } catch (err) {
        isPyodideReady = false;
        showError(`Failed to initialize Python environment: ${err}`);
        setLoadingState(false);
    }
}

function setLoadingState(isLoading, message = '') {
    statusMessage.textContent = message;
    statusDiv.style.display = isLoading ? 'flex' : 'none';
    runButton.disabled = isLoading;
    runButton.classList.toggle('btn-disabled', isLoading);
    if (!isLoading) updateButtonState();
}

function showError(message) {
    errorMessage.textContent = message;
    errorBox.classList.remove('hidden');
    outputDiv.classList.add('hidden');
}

function hideError() {
    errorBox.classList.add('hidden');
}

function createParameterControl(paramName, paramValue, paramLabel, isCompact = false) {
    const container = document.createElement('div');
    
    if (isCompact) {
        // Compact layout - just label and input in a flex row
        container.className = 'flex items-center justify-between py-1';
        
        const label = document.createElement('label');
        label.setAttribute('for', `${paramName}_input`);
        label.className = 'text-xs font-medium text-gray-700 flex-shrink-0 w-16';
        label.innerHTML = paramLabel;
        
        const inputContainer = document.createElement('div');
        inputContainer.className = 'flex items-center space-x-1 flex-grow';
        
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `${paramName}_input`;
        input.className = 'w-20 px-2 py-1 border border-gray-300 rounded text-xs text-center';
        input.step = 'any';
        
        const resetBtn = document.createElement('button');
        resetBtn.className = 'px-1 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded border';
        resetBtn.innerHTML = '↺';
        resetBtn.title = 'Reset to fitted value';
        resetBtn.type = 'button';
        
        inputContainer.appendChild(input);
        inputContainer.appendChild(resetBtn);
        container.appendChild(label);
        container.appendChild(inputContainer);
        
        return { container, slider: null, input, resetBtn };
    } else {
        // Original layout with sliders
        const label = document.createElement('label');
        label.setAttribute('for', `${paramName}_slider`);
        label.className = 'block text-sm font-medium text-gray-700';
        label.innerHTML = paramLabel;
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = `${paramName}_slider`;
        slider.className = 'mt-1 w-full';
        
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `${paramName}_input`;
        input.className = 'mt-2 w-full p-2 border border-gray-300 rounded-md text-sm';
        input.step = 'any';
        
        container.appendChild(label);
        container.appendChild(slider);
        container.appendChild(input);
        
        return { container, slider, input };
    }
}

function createDynamicParameterControls(modelKey, fittedParams) {
    const dynamicContainer = document.getElementById('dynamic-parameters');
    dynamicContainer.innerHTML = ''; // Clear existing controls
    paramControls = {}; // Reset parameter controls
    
    if (modelKey === 'djordjevic_sarkar') {
        // Create Djordjevic-Sarkar controls
        const controls = {
            eps_inf: createParameterControl('eps_inf', fittedParams.eps_inf, 'ε<sub>∞</sub> (Epsilon Infinity)'),
            delta_eps: createParameterControl('delta_eps', fittedParams.delta_eps, 'Δε (Delta Epsilon)'),
            omega1: createParameterControl('omega1', fittedParams.omega1, 'f<sub>1</sub> (GHz)'),
            omega2: createParameterControl('omega2', fittedParams.omega2, 'f<sub>2</sub> (GHz)')
        };
        
        for (const [key, control] of Object.entries(controls)) {
            dynamicContainer.appendChild(control.container);
            paramControls[key] = { slider: control.slider, input: control.input };
        }
    } else if (modelKey === 'hybrid_debye_lorentz') {
        // Create Hybrid Debye-Lorentz controls with compact layout
        const controls = {};
        
        // Find number of terms
        const nTerms = Object.keys(fittedParams).filter(key => key.startsWith('delta_eps_')).length;
        
        // Add header
        const headerDiv = document.createElement('div');
        headerDiv.className = 'mb-3 p-2 bg-blue-50 rounded-lg';
        headerDiv.innerHTML = `<h4 class="text-sm font-semibold text-blue-800">Hybrid Debye-Lorentz Model (${nTerms} terms)</h4>`;
        dynamicContainer.appendChild(headerDiv);
        
        // eps_inf control (slightly larger since it's important)
        const epsInfContainer = document.createElement('div');
        epsInfContainer.className = 'mb-3 p-2 bg-gray-50 rounded';
        controls.eps_inf = createParameterControl('eps_inf', fittedParams.eps_inf, 'ε<sub>∞</sub>', true);
        controls.eps_inf.container.className = 'flex items-center justify-between py-2';
        controls.eps_inf.container.querySelector('label').className = 'text-sm font-medium text-gray-700 flex-shrink-0 w-20';
        epsInfContainer.appendChild(controls.eps_inf.container);
        dynamicContainer.appendChild(epsInfContainer);
        paramControls.eps_inf = { slider: controls.eps_inf.slider, input: controls.eps_inf.input, resetBtn: controls.eps_inf.resetBtn };
        
        // Create a grid container for all terms
        const termsContainer = document.createElement('div');
        termsContainer.className = 'space-y-2';
        
        // Create controls for each term
        for (let i = 0; i < nTerms; i++) {
            const deltaKey = `delta_eps_${i}`;
            const fKey = `f_k_${i}`;
            const sigmaKey = `sigma_k_${i}`;
            
            // Create term container
            const termContainer = document.createElement('div');
            termContainer.className = 'border border-gray-200 rounded-lg p-2 bg-white';
            
            const termHeader = document.createElement('div');
            termHeader.className = 'text-xs font-semibold text-gray-600 mb-2 border-b border-gray-200 pb-1';
            termHeader.textContent = `Term ${i + 1}`;
            termContainer.appendChild(termHeader);
            
            // Create compact controls
            controls[deltaKey] = createParameterControl(deltaKey, fittedParams[deltaKey], `Δε<sub>${i}</sub>`, true);
            controls[fKey] = createParameterControl(fKey, fittedParams[fKey], `f<sub>${i}</sub>`, true);
            controls[sigmaKey] = createParameterControl(sigmaKey, fittedParams[sigmaKey], `σ<sub>${i}</sub>`, true);
            
            termContainer.appendChild(controls[deltaKey].container);
            termContainer.appendChild(controls[fKey].container);
            termContainer.appendChild(controls[sigmaKey].container);
            
            termsContainer.appendChild(termContainer);
            
            // Store controls
            paramControls[deltaKey] = { slider: controls[deltaKey].slider, input: controls[deltaKey].input, resetBtn: controls[deltaKey].resetBtn };
            paramControls[fKey] = { slider: controls[fKey].slider, input: controls[fKey].input, resetBtn: controls[fKey].resetBtn };
            paramControls[sigmaKey] = { slider: controls[sigmaKey].slider, input: controls[sigmaKey].input, resetBtn: controls[sigmaKey].resetBtn };
        }
        
        dynamicContainer.appendChild(termsContainer);
    }
    
    // Set up event listeners for the new controls
    setupParameterEventListeners();
}

function updateButtonState() {
    const isReady = isPyodideReady && !!fileContent;
    runButton.disabled = !isReady;
    runButton.classList.toggle('btn-disabled', !isReady);
}

function setupParameterEventListeners() {
    let updateTimeout;
    
    Object.entries(paramControls).forEach(([key, controlObj]) => {
        const { slider, input, resetBtn } = controlObj;
        
        // Remove existing listeners to avoid duplicates
        if (slider) {
            slider.removeEventListener('input', slider._inputHandler);
        }
        input.removeEventListener('input', input._inputHandler);
        if (resetBtn) {
            resetBtn.removeEventListener('click', resetBtn._clickHandler);
        }
        
        // Create input handler
        input._inputHandler = (e) => {
            if (slider) {
                slider.value = e.target.value;
            }
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(updateUIFromControls, 50);
        };
        
        // Add input listener
        input.addEventListener('input', input._inputHandler);
        
        // Create slider handler if slider exists
        if (slider) {
            slider._inputHandler = (e) => {
                input.value = e.target.value;
                clearTimeout(updateTimeout);
                updateTimeout = setTimeout(updateUIFromControls, 50);
            };
            slider.addEventListener('input', slider._inputHandler);
        }
        
        // Create reset button handler if it exists
        if (resetBtn) {
            resetBtn._clickHandler = (e) => {
                e.preventDefault();
                // Reset to original fitted value (stored when we set up the controls)
                if (currentData && currentData.fitted_params && currentData.fitted_params[key] !== undefined) {
                    const originalValue = currentData.fitted_params[key];
                    input.value = originalValue;
                    if (slider) {
                        slider.value = originalValue;
                    }
                    clearTimeout(updateTimeout);
                    updateTimeout = setTimeout(updateUIFromControls, 50);
                }
            };
            resetBtn.addEventListener('click', resetBtn._clickHandler);
        }
    });
}

function setupParameterControls(modelKey, fittedParams) {
    // Create dynamic controls first
    createDynamicParameterControls(modelKey, fittedParams);
    
    // Set up ranges and values
    if (modelKey === 'djordjevic_sarkar') {
        const twoPi = 2 * Math.PI;
        const f1_ghz = fittedParams.omega1 / (twoPi * 1e9);
        const f2_ghz = fittedParams.omega2 / (twoPi * 1e9);

        const uiParams = {
            eps_inf: fittedParams.eps_inf,
            delta_eps: fittedParams.delta_eps,
            omega1: f1_ghz,
            omega2: f2_ghz
        };

        const ranges = {
            eps_inf: { min: 1, max: Math.max(uiParams.eps_inf * 2, 10), step: 0.01 },
            delta_eps: { min: 0, max: Math.max(uiParams.delta_eps * 2, 5), step: 0.01 },
            omega1: { min: uiParams.omega1 / 10, max: uiParams.omega1 * 10, step: uiParams.omega1 / 100 },
            omega2: { min: uiParams.omega2 / 10, max: uiParams.omega2 * 10, step: uiParams.omega2 / 100 },
        };

        for (const key in uiParams) {
            if (paramControls[key]) {
                const { slider, input } = paramControls[key];
                const range = ranges[key];

                slider.min = range.min;
                slider.max = range.max;
                slider.step = range.step;
                slider.value = uiParams[key];

                input.min = range.min;
                input.max = range.max;
                input.step = range.step;
                input.value = uiParams[key];
            }
        }
    } else if (modelKey === 'hybrid_debye_lorentz') {
        // Set up values for hybrid model parameters (compact layout - no sliders)
        for (const [key, value] of Object.entries(fittedParams)) {
            if (paramControls[key]) {
                const { slider, input } = paramControls[key];
                
                // Set step and constraints for input
                if (key === 'eps_inf') {
                    input.min = 1;
                    input.step = 0.01;
                } else if (key.startsWith('delta_eps_')) {
                    input.min = 0.001;
                    input.step = 0.01;
                } else if (key.startsWith('f_k_')) {
                    input.min = 0.001;
                    input.step = 0.001;
                } else if (key.startsWith('sigma_k_')) {
                    input.min = 0.0001;
                    input.step = 0.0001;
                }
                
                // Set the initial value
                input.value = value;
                
                // Set slider value if it exists (for Djordjevic-Sarkar compatibility)
                if (slider) {
                    slider.value = value;
                }
            }
        }
    }
}

async function updateUIFromControls() {
    if (!currentData || !pyodide) return;

    try {
        let modelResults, reportData, evaluationData;
        
        if (currentModel === 'djordjevic_sarkar') {
            // Handle Djordjevic-Sarkar model
            const eps_inf = parseFloat(paramControls.eps_inf.input.value);
            const delta_eps = parseFloat(paramControls.delta_eps.input.value);
            const f1_ghz = parseFloat(paramControls.omega1.input.value);
            const f2_ghz = parseFloat(paramControls.omega2.input.value);

            const twoPi = 2 * Math.PI;
            const omega1 = f1_ghz * twoPi * 1e9;
            const omega2 = f2_ghz * twoPi * 1e9;

            // Update Plot
            const calculateFunc = pyodide.globals.get('calculate_model_from_params');
            const modelJson = calculateFunc(eps_inf, delta_eps, omega1, omega2);
            modelResults = JSON.parse(modelJson);

            // Update Report
            const generateReportFunc = pyodide.globals.get('create_updated_report');
            const reportJson = generateReportFunc(eps_inf, delta_eps, omega1, omega2);
            reportData = JSON.parse(reportJson);

            // Update fit quality indicator
            const evaluateFunc = pyodide.globals.get('evaluate_current_fit');
            const evaluationJson = evaluateFunc(eps_inf, delta_eps, omega1, omega2);
            evaluationData = JSON.parse(evaluationJson);
            
        } else if (currentModel === 'hybrid_debye_lorentz') {
            // Handle Hybrid Debye-Lorentz model
            const currentParams = {};
            
            // Collect all parameter values
            Object.keys(paramControls).forEach(key => {
                currentParams[key] = parseFloat(paramControls[key].input.value);
            });
            
            // Use generic calculation function
            const calculateGenericFunc = pyodide.globals.get('calculate_model_from_params_generic');
            const paramsPython = pyodide.toPy(currentParams);
            const fGhzPython = pyodide.toPy(currentData.f_ghz);
            const modelJson = calculateGenericFunc(currentModel, paramsPython, fGhzPython);
            modelResults = JSON.parse(modelJson);
            
            // For now, skip report and evaluation updates for hybrid model
            // TODO: Implement generic report and evaluation functions
            reportData = { report: "Report generation for Hybrid Debye-Lorentz model coming soon..." };
            evaluationData = { evaluation: null };
        }

        if (modelResults.error) {
            console.error('Model calculation error:', modelResults.error);
            return;
        }

        // Update plot
        const fittedDk = modelResults.eps_prime;
        const fittedDf = fittedDk.map((dk, i) => -modelResults.eps_double_prime[i] / dk);
        Plotly.restyle('plot-output', { y: [fittedDk, fittedDf] }, [1, 3]);

        // Update report
        if (reportData.report) {
            reportOutput.textContent = reportData.report;
        }

        // Update fit quality indicator
        if (evaluationData.evaluation) {
            updateFitQualityIndicator(evaluationData.evaluation);
        } else {
            // Hide fit quality indicator for unsupported models
            document.getElementById('fit-quality-indicator').classList.add('hidden');
        }

    } catch (err) {
        console.error('Error updating UI:', err);
    }
}

function updateFitQualityIndicator(evaluation) {
    const indicator = document.getElementById('fit-quality-indicator');
    const scoreSpan = document.getElementById('fit-quality-score');
    const bar = document.getElementById('fit-quality-bar');

    if (!evaluation) {
        indicator.classList.add('hidden');
        return;
    }

    indicator.classList.remove('hidden');
    const score = evaluation.overall_score;
    const category = evaluation.overall_category;

    scoreSpan.textContent = `${score} (${category})`;

    // Update bar width and color
    bar.style.width = `${score}%`;

    if (score >= 90) {
        bar.className = 'h-2 rounded-full transition-all duration-300 bg-green-500';
        indicator.className = 'mb-4 p-3 rounded-lg bg-green-50 border border-green-200';
    } else if (score >= 75) {
        bar.className = 'h-2 rounded-full transition-all duration-300 bg-blue-500';
        indicator.className = 'mb-4 p-3 rounded-lg bg-blue-50 border border-blue-200';
    } else if (score >= 60) {
        bar.className = 'h-2 rounded-full transition-all duration-300 bg-yellow-500';
        indicator.className = 'mb-4 p-3 rounded-lg bg-yellow-50 border border-yellow-200';
    } else {
        bar.className = 'h-2 rounded-full transition-all duration-300 bg-red-500';
        indicator.className = 'mb-4 p-3 rounded-lg bg-red-50 border border-red-200';
    }
}

function resetParameters() {
    if (!currentData || !currentData.fitted_params) return;
    
    setupParameterControls(currentModel, currentData.fitted_params);

    const plotData = JSON.parse(currentData.plot_json);
    Plotly.restyle('plot-output', {
        y: [plotData.data[1].y, plotData.data[3].y]
    }, [1, 3]);
    reportOutput.textContent = currentData.report;

    // Update fit quality for reset parameters (only for Djordjevic-Sarkar)
    if (currentModel === 'djordjevic_sarkar') {
        const evaluateFunc = pyodide.globals.get('evaluate_current_fit');
        const params = currentData.fitted_params;
        const evaluationJson = evaluateFunc(params.eps_inf, params.delta_eps, params.omega1, params.omega2);
        const evaluationData = JSON.parse(evaluationJson);

        if (evaluationData.evaluation) {
            updateFitQualityIndicator(evaluationData.evaluation);
        }
    }
}

async function downloadResults() {
    if (!currentData) return;
    setLoadingState(true, 'Preparing download...');
    try {
        const zip = new JSZip();
        const baseFilename = originalFilename.replace(/\.csv$/i, '');

        const eps_inf = parseFloat(paramControls.eps_inf.input.value);
        const delta_eps = parseFloat(paramControls.delta_eps.input.value);
        const f1_ghz = parseFloat(paramControls.omega1.input.value);
        const f2_ghz = parseFloat(paramControls.omega2.input.value);
        const twoPi = 2 * Math.PI;
        const omega1 = f1_ghz * twoPi * 1e9;
        const omega2 = f2_ghz * twoPi * 1e9;

        // 1. Create Fitted Data CSV
        const calculateFunc = pyodide.globals.get('calculate_model_from_params');
        const modelJson = calculateFunc(eps_inf, delta_eps, omega1, omega2);
        const modelResults = JSON.parse(modelJson);

        let csvContent = "Frequency_GHz,fitted_Dk,fitted_Df\n";
        currentData.f_ghz.forEach((f, i) => {
            const dk = modelResults.eps_prime[i];
            const df = dk === 0 ? 0 : -modelResults.eps_double_prime[i] / dk;
            csvContent += `${f},${dk},${df}\n`;
        });
        zip.file(`${baseFilename}_fitted.csv`, csvContent);

        // 2. Add Matplotlib Plot Image
        const createPlotFunc = pyodide.globals.get('create_downloadable_plot');
        const imageProxy = createPlotFunc(eps_inf, delta_eps, omega1, omega2);
        const imageData = imageProxy.toJs();
        imageProxy.destroy();
        zip.file(`${baseFilename}_plot.png`, imageData);

        // 3. Add Report
        zip.file(`${baseFilename}_report.txt`, reportOutput.textContent);

        // 4. Generate JSON report
        const generateReportFunc = pyodide.globals.get('create_updated_report');
        const reportJson = generateReportFunc(eps_inf, delta_eps, omega1, omega2);
        const reportData = JSON.parse(reportJson);

        // Get evaluation
        const evaluateFunc = pyodide.globals.get('evaluate_current_fit');
        const evaluationJson = evaluateFunc(eps_inf, delta_eps, omega1, omega2);
        const evaluationData = JSON.parse(evaluationJson);

        // Get measured data
        let measuredDk = currentData.measured_dk || [];
        let measuredDf = currentData.measured_df || [];

        // Add additional fields to JSON
        const fullJsonReport = {
            ...reportData.json_data,
            input_file: originalFilename,
            data: {
                frequency_ghz: currentData.f_ghz,
                measured: {
                    dk: measuredDk,
                    df: measuredDf
                },
                fitted: {
                    dk: modelResults.eps_prime,
                    df: modelResults.eps_prime.map((dk, i) => {
                        return dk === 0 ? 0 : -modelResults.eps_double_prime[i] / dk;
                    })
                }
            },
            evaluation: evaluationData.evaluation
        };

        zip.file(`${baseFilename}_report.json`, JSON.stringify(fullJsonReport, null, 2));

        // 5. Add evaluation report
        if (evaluationData.evaluation && evaluationData.evaluation.markdown_table) {
            zip.file(`${baseFilename}_evaluation.md`, evaluationData.evaluation.markdown_table);
        }

        // Generate and Download Zip
        const content = await zip.generateAsync({ type: "blob" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(content);
        link.download = `${baseFilename}_results.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

    } catch(err) {
        showError(`Failed to create download package: ${err}`);
    } finally {
        setLoadingState(false);
    }
}

// Dynamic parameter event listeners are handled in setupParameterEventListeners()

fileUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        originalFilename = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            fileContent = e.target.result;
            updateButtonState();
        };
        reader.readAsText(file);
    } else {
        fileContent = null;
        originalFilename = '';
        updateButtonState();
    }
});

// Model selection handler
modelSelect.addEventListener('change', (e) => {
    const selectedModel = e.target.value;
    if (selectedModel === 'hybrid_debye_lorentz') {
        modelConfig.classList.remove('hidden');
    } else {
        modelConfig.classList.add('hidden');
    }
});

runButton.addEventListener('click', async () => {
    if (!fileContent || !pyodide) return;
    hideError();
    setLoadingState(true, 'Running analysis...');
    outputDiv.classList.add('hidden');

    try {
        const selectedModel = modelSelect.value;
        const fitMethod = fitMethodSelect.value;
        
        console.log(`Running analysis with model: ${selectedModel}, method: ${fitMethod}`);
        
        let resultJson;
        if (selectedModel === 'djordjevic_sarkar') {
            // Use legacy function for backward compatibility
            const runAnalysis = pyodide.globals.get('run_analysis');
            resultJson = runAnalysis(fileContent, fitMethod);
        } else {
            // Use new model-aware function
            console.log('Getting run_analysis_with_model function...');
            const runAnalysisWithModel = pyodide.globals.get('run_analysis_with_model');
            if (!runAnalysisWithModel) {
                throw new Error('run_analysis_with_model function not found in Python globals');
            }
            
            console.log('Converting model config to Python...');
            // Convert JavaScript object to Python dict
            const modelConfigPython = pyodide.toPy({
                n_terms: parseInt(nTermsSelect.value)
            });
            
            console.log('Calling Python function...');
            resultJson = runAnalysisWithModel(fileContent, selectedModel, modelConfigPython, fitMethod);
        }
        
        const result = JSON.parse(resultJson);

        if (result.error) {
            throw new Error(result.error);
        }

        currentData = result;
        currentModel = result.model_key || selectedModel;

        const plotData = JSON.parse(result.plot_json);
        plotOutput.innerHTML = '';
        Plotly.newPlot('plot-output', plotData.data, plotData.layout, {responsive: true});

        // Store measured data
        currentData.measured_dk = result.measured_dk;
        currentData.measured_df = result.measured_df;

        reportOutput.textContent = result.report;
        setupParameterControls(currentModel, result.fitted_params);

        // Show initial fit quality (only for Djordjevic-Sarkar for now)
        if (currentModel === 'djordjevic_sarkar') {
            const evaluateFunc = pyodide.globals.get('evaluate_current_fit');
            const eps_inf = result.fitted_params.eps_inf;
            const delta_eps = result.fitted_params.delta_eps;
            const omega1 = result.fitted_params.omega1;
            const omega2 = result.fitted_params.omega2;

            const evaluationJson = evaluateFunc(eps_inf, delta_eps, omega1, omega2);
            const evaluationData = JSON.parse(evaluationJson);

            if (evaluationData.evaluation) {
                updateFitQualityIndicator(evaluationData.evaluation);
            }
        } else {
            // Hide fit quality indicator for other models for now
            document.getElementById('fit-quality-indicator').classList.add('hidden');
        }

        outputDiv.classList.remove('hidden');

    } catch (err) {
        console.error(err);
        showError(`An error occurred during analysis: ${err.message}`);
    } finally {
        setLoadingState(false);
    }
});

resetButton.addEventListener('click', resetParameters);
downloadButton.addEventListener('click', downloadResults);

// Initialize
main();
updateButtonState();