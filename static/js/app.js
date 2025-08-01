// DOM Elements
const fileUpload = document.getElementById('file-upload');
const runButton = document.getElementById('run-button');
const resetButton = document.getElementById('reset-button');
const downloadButton = document.getElementById('download-button');
const fitMethodSelect = document.getElementById('fit-method');
const statusDiv = document.getElementById('status');
const statusMessage = document.getElementById('status-message');
const outputDiv = document.getElementById('output');
const plotOutput = document.getElementById('plot-output');
const reportOutput = document.getElementById('report-output');
const errorBox = document.getElementById('error-box');
const errorMessage = document.getElementById('error-message');

// Parameter Controls
const paramControls = {
    eps_inf: { slider: document.getElementById('eps_inf_slider'), input: document.getElementById('eps_inf_input') },
    delta_eps: { slider: document.getElementById('delta_eps_slider'), input: document.getElementById('delta_eps_input') },
    omega1: { slider: document.getElementById('omega1_slider'), input: document.getElementById('omega1_input') },
    omega2: { slider: document.getElementById('omega2_slider'), input: document.getElementById('omega2_input') },
};

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
    create_downloadable_plot, get_measured_data
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

function updateButtonState() {
    const isReady = isPyodideReady && !!fileContent;
    runButton.disabled = !isReady;
    runButton.classList.toggle('btn-disabled', !isReady);
}

function setupParameterControls(params) {
    const twoPi = 2 * Math.PI;
    const f1_ghz = params.omega1 / (twoPi * 1e9);
    const f2_ghz = params.omega2 / (twoPi * 1e9);

    const uiParams = {
        eps_inf: params.eps_inf,
        delta_eps: params.delta_eps,
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
}

async function updateUIFromControls() {
    if (!currentData || !pyodide) return;

    const eps_inf = parseFloat(paramControls.eps_inf.input.value);
    const delta_eps = parseFloat(paramControls.delta_eps.input.value);
    const f1_ghz = parseFloat(paramControls.omega1.input.value);
    const f2_ghz = parseFloat(paramControls.omega2.input.value);

    const twoPi = 2 * Math.PI;
    const omega1 = f1_ghz * twoPi * 1e9;
    const omega2 = f2_ghz * twoPi * 1e9;

    try {
        // Update Plot
        const calculateFunc = pyodide.globals.get('calculate_model_from_params');
        const modelJson = calculateFunc(eps_inf, delta_eps, omega1, omega2);
        const modelResults = JSON.parse(modelJson);

        if (modelResults.error) return;

        const fittedDk = modelResults.eps_prime;
        const fittedDf = fittedDk.map((dk, i) => -modelResults.eps_double_prime[i] / dk);
        Plotly.restyle('plot-output', { y: [fittedDk, fittedDf] }, [1, 3]);

        // Update Report
        const generateReportFunc = pyodide.globals.get('create_updated_report');
        const reportJson = generateReportFunc(eps_inf, delta_eps, omega1, omega2);
        const reportData = JSON.parse(reportJson);
        reportOutput.textContent = reportData.report;

        // Update fit quality indicator
        const evaluateFunc = pyodide.globals.get('evaluate_current_fit');
        const evaluationJson = evaluateFunc(eps_inf, delta_eps, omega1, omega2);
        const evaluationData = JSON.parse(evaluationJson);

        if (evaluationData.evaluation) {
            updateFitQualityIndicator(evaluationData.evaluation);
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
    setupParameterControls(currentData.fitted_params);

    const plotData = JSON.parse(currentData.plot_json);
    Plotly.restyle('plot-output', {
        y: [plotData.data[1].y, plotData.data[3].y]
    }, [1, 3]);
    reportOutput.textContent = currentData.report;

    // Update fit quality for reset parameters
    const evaluateFunc = pyodide.globals.get('evaluate_current_fit');
    const params = currentData.fitted_params;
    const evaluationJson = evaluateFunc(params.eps_inf, params.delta_eps, params.omega1, params.omega2);
    const evaluationData = JSON.parse(evaluationJson);

    if (evaluationData.evaluation) {
        updateFitQualityIndicator(evaluationData.evaluation);
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

// Set up event listeners
let updateTimeout;
Object.entries(paramControls).forEach(([key, { slider, input }]) => {
    slider.addEventListener('input', (e) => {
        input.value = e.target.value;
        clearTimeout(updateTimeout);
        updateTimeout = setTimeout(updateUIFromControls, 50);
    });

    input.addEventListener('input', (e) => {
        slider.value = e.target.value;
        clearTimeout(updateTimeout);
        updateTimeout = setTimeout(updateUIFromControls, 50);
    });
});

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

runButton.addEventListener('click', async () => {
    if (!fileContent || !pyodide) return;
    hideError();
    setLoadingState(true, 'Running analysis...');
    outputDiv.classList.add('hidden');

    try {
        const runAnalysis = pyodide.globals.get('run_analysis');
        const fitMethod = fitMethodSelect.value;
        const resultJson = runAnalysis(fileContent, fitMethod);
        const result = JSON.parse(resultJson);

        if (result.error) {
            throw new Error(result.error);
        }

        currentData = result;

        const plotData = JSON.parse(result.plot_json);
        plotOutput.innerHTML = '';
        Plotly.newPlot('plot-output', plotData.data, plotData.layout, {responsive: true});

        // Store measured data
        currentData.measured_dk = result.measured_dk;
        currentData.measured_df = result.measured_df;

        reportOutput.textContent = result.report;
        setupParameterControls(result.fitted_params);

        // Show initial fit quality
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