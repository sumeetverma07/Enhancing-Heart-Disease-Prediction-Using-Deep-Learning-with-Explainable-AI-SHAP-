const schema = window.APP_SCHEMA || [];
const form = document.getElementById("patient-form");
const downloadButton = document.getElementById("download-report");
const statusPill = document.getElementById("status-pill");
const apiStatus = document.getElementById("api-status");
const explanationSelect = document.getElementById("explanation-model-select");

const elements = {
    primaryModel: document.getElementById("primary-model"),
    explanationModel: document.getElementById("explanation-model"),
    timestamp: document.getElementById("prediction-timestamp"),
    riskProbability: document.getElementById("risk-probability"),
    riskLabel: document.getElementById("risk-label"),
    confidenceScore: document.getElementById("confidence-score"),
    annProbability: document.getElementById("ann-probability"),
    importanceBody: document.getElementById("importance-body"),
};

const state = {
    debounceTimer: null,
    latestResponse: null,
    explanationModel: "Random Forest",
    requestController: null,
    requestSequence: 0,
};

function setStatus(text, apiText, tone = "idle") {
    statusPill.textContent = text;
    apiStatus.textContent = apiText;
    statusPill.style.background = tone === "error" ? "#fdeaea" : tone === "success" ? "#e9f7ef" : "#f7efe4";
    statusPill.style.color = tone === "error" ? "#9f2d2d" : tone === "success" ? "#186448" : "#8a5a20";
}

function getPayload() {
    const payload = {};
    schema.forEach((field) => {
        const element = form.elements.namedItem(field.name);
        if (!element) {
            return;
        }
        if (field.type === "slider_int") {
            payload[field.name] = Number.parseInt(element.value, 10);
        } else if (field.type === "slider_float") {
            payload[field.name] = Number.parseFloat(element.value);
        } else {
            payload[field.name] = element.value;
        }
    });
    payload.explanation_model = state.explanationModel;
    return payload;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function renderErrors(errors = {}) {
    schema.forEach((field) => {
        const errorNode = document.querySelector(`[data-error-for="${field.name}"]`);
        if (errorNode) {
            errorNode.textContent = errors[field.name] || "";
        }
    });
}

function renderMetrics(response) {
    const prediction = response.prediction;
    if (response.available_explanations) {
        explanationSelect.innerHTML = response.available_explanations.map((name) => `
            <option value="${name}" ${name === response.explanation_model ? "selected" : ""}>${name}</option>
        `).join("");
        state.explanationModel = response.explanation_model;
    }
    elements.primaryModel.textContent = prediction.primary_model;
    elements.explanationModel.textContent = response.explanation_model;
    elements.timestamp.textContent = response.timestamp;
    elements.riskProbability.textContent = `${(prediction.probability * 100).toFixed(1)}%`;
    elements.riskLabel.textContent = prediction.label;
    elements.confidenceScore.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
    const annProbability = prediction.all_models.ANN;
    elements.annProbability.textContent = annProbability !== undefined ? `${(annProbability * 100).toFixed(1)}%` : "N/A";
    apiStatus.textContent = `Live • ${response.inference_time_ms.toFixed(0)} ms`;
}

function renderModelChart(probabilities) {
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities).map((value) => value * 100);
    Plotly.react("model-chart", [
        {
            type: "bar",
            x: labels,
            y: values,
            marker: {
                color: ["#1f4e79", "#4d7ea8", "#cc5a3c"].slice(0, labels.length),
                line: { color: "#102542", width: 0.6 },
            },
            hovertemplate: "%{x}<br>%{y:.2f}%<extra></extra>",
        },
    ], {
        margin: { l: 50, r: 10, t: 10, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        yaxis: { title: "Probability (%)", rangemode: "tozero" },
        xaxis: { title: "" },
    }, { displayModeBar: false, responsive: true });
}

function renderShapChart(shapValues) {
    const ordered = [...shapValues].reverse();
    Plotly.react("shap-chart", [
        {
            type: "bar",
            orientation: "h",
            y: ordered.map((item) => item.feature),
            x: ordered.map((item) => item.shap_value),
            text: ordered.map((item) => item.direction),
            marker: {
                color: ordered.map((item) => item.shap_value >= 0 ? "#cc5a3c" : "#1f4e79"),
            },
            hovertemplate: "%{y}<br>SHAP: %{x:.5f}<br>%{text}<extra></extra>",
        },
    ], {
        margin: { l: 90, r: 20, t: 10, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "SHAP value" },
        yaxis: { automargin: true },
    }, { displayModeBar: false, responsive: true });
}

function renderImportanceTable(rows) {
    elements.importanceBody.innerHTML = rows.map((row) => `
        <tr>
            <td>${escapeHtml(row.feature)}</td>
            <td>${escapeHtml(row.value)}</td>
            <td>${row.shap_value.toFixed(5)}</td>
            <td>${escapeHtml(row.direction)}</td>
        </tr>
    `).join("");
}

function renderAnnChart(annData) {
    if (!annData) {
        Plotly.react("ann-chart", [], {
            annotations: [{
                text: "ANN visualization is not available in the current runtime.",
                showarrow: false,
                font: { size: 16, color: "#55687a" },
            }],
            xaxis: { visible: false },
            yaxis: { visible: false },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
        }, { displayModeBar: false, responsive: true });
        return;
    }

    const layerCount = annData.layers.length;
    const traces = [];
    const annotations = [];
    const edgeShapes = [];
    const nodePositions = new Map();

    annData.layers.forEach((layer, layerIndex) => {
        const x = layerCount === 1 ? 0.5 : layerIndex / (layerCount - 1);
        const sortedNodes = [...layer.nodes].sort((a, b) => Math.abs(b.normalized) - Math.abs(a.normalized));
        const nodeCount = sortedNodes.length;
        const yPositions = sortedNodes.map((_, nodeIndex) => nodeCount === 1 ? 0.5 : 1 - (nodeIndex / (nodeCount - 1)));
        sortedNodes.forEach((node, nodeIndex) => {
            nodePositions.set(node.id, { x, y: yPositions[nodeIndex], node });
        });

        traces.push({
            type: "scatter",
            mode: "markers+text",
            x: Array(nodeCount).fill(x),
            y: yPositions,
            text: sortedNodes.map((node) => `${node.label}<br>${node.value}`),
            textposition: layerIndex === 0 ? "middle left" : layerIndex === layerCount - 1 ? "middle right" : "top center",
            hovertemplate: "%{text}<extra></extra>",
            marker: {
                size: sortedNodes.map((node) => 18 + Math.abs(node.normalized) * 16),
                color: sortedNodes.map((node) => node.normalized),
                colorscale: [
                    [0, "#1f4e79"],
                    [0.5, "#e8eef5"],
                    [1, "#cc5a3c"],
                ],
                cmin: -1,
                cmax: 1,
                line: { color: "#102542", width: 1 },
            },
            showlegend: false,
        });

        annotations.push({
            x,
            y: 1.08,
            xref: "x",
            yref: "paper",
            text: layer.name,
            showarrow: false,
            font: { size: 13, color: "#102542" },
        });
    });

    annData.connections.forEach((connection) => {
        const from = nodePositions.get(connection.from_id);
        const to = nodePositions.get(connection.to_id);
        if (!from || !to) {
            return;
        }
        edgeShapes.push({
            type: "line",
            x0: from.x,
            y0: from.y,
            x1: to.x,
            y1: to.y,
            line: {
                color: connection.weight >= 0 ? "rgba(204,90,60,0.55)" : "rgba(31,78,121,0.55)",
                width: 1 + Math.abs(connection.normalized_weight) * 4,
            },
            layer: "below",
        });
    });

    Plotly.react("ann-chart", traces, {
        margin: { l: 40, r: 40, t: 30, b: 20 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { visible: false, range: [-0.08, 1.08] },
        yaxis: { visible: false, range: [-0.05, 1.05] },
        shapes: edgeShapes,
        annotations,
    }, { displayModeBar: false, responsive: true });
}

async function requestPrediction() {
    setStatus("Refreshing model outputs...", "Loading", "idle");
    apiStatus.textContent = "Loading";
    renderErrors();
    if (state.requestController) {
        state.requestController.abort();
    }
    state.requestController = new AbortController();
    const requestId = ++state.requestSequence;
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getPayload()),
            signal: state.requestController.signal,
        });
        const data = await response.json();
        if (requestId !== state.requestSequence) {
            return;
        }
        if (!response.ok) {
            renderErrors(data.errors || {});
            throw new Error(data.message || "Prediction failed.");
        }

        state.latestResponse = data;
        renderMetrics(data);
        renderModelChart(data.prediction.all_models);
        renderShapChart(data.shap_values);
        renderImportanceTable(data.feature_importance);
        renderAnnChart(data.ann_visualization);
        setStatus("Prediction updated", "Live", "success");
    } catch (error) {
        if (error.name === "AbortError") {
            return;
        }
        console.error(error);
        setStatus(error.message, "Error", "error");
        apiStatus.textContent = "Error";
    } finally {
        if (requestId === state.requestSequence) {
            state.requestController = null;
        }
    }
}

function queuePrediction() {
    window.clearTimeout(state.debounceTimer);
    state.debounceTimer = window.setTimeout(requestPrediction, 320);
}

async function downloadReport() {
    downloadButton.disabled = true;
    setStatus("Preparing PDF report...", "Exporting", "idle");
    try {
        const response = await fetch("/download-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getPayload()),
        });
        if (!response.ok) {
            const data = await response.json();
            renderErrors(data.errors || {});
            throw new Error(data.message || "Unable to generate report.");
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "heart_disease_prediction_report.pdf";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.URL.revokeObjectURL(url);
        setStatus("Report downloaded", "Ready", "success");
    } catch (error) {
        console.error(error);
        setStatus(error.message, "Error", "error");
    } finally {
        downloadButton.disabled = false;
    }
}

function bindForm() {
    schema.forEach((field) => {
        const input = form.elements.namedItem(field.name);
        const output = document.querySelector(`[data-output-for="${field.name}"]`);
        if (!input) {
            return;
        }

        input.addEventListener("input", () => {
            if (output) {
                output.textContent = input.value;
            }
            queuePrediction();
        });
        input.addEventListener("change", queuePrediction);
    });
    explanationSelect.addEventListener("change", () => {
        state.explanationModel = explanationSelect.value;
        queuePrediction();
    });
    downloadButton.addEventListener("click", downloadReport);
}

bindForm();
requestPrediction();
