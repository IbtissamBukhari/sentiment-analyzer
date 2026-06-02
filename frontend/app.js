const API_URL = "http://127.0.0.1:8000";

function showLoading() {
    document.getElementById("loading")
        .classList.remove("hidden");
}

function hideLoading() {
    document.getElementById("loading")
        .classList.add("hidden");
}

async function predictSentiment() {

    const text =
        document.getElementById("textInput").value;

    if (!text.trim()) {
        alert("Please enter text.");
        return;
    }

    showLoading();

    try {

        const response =
            await fetch(
                `${API_URL}/predict`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        text
                    })
                }
            );

        const data =
            await response.json();

        let sentimentClass = "";

        if (data.label === "positive")
            sentimentClass =
                "sentiment-positive";

        else if (data.label === "negative")
            sentimentClass =
                "sentiment-negative";

        else
            sentimentClass =
                "sentiment-neutral";

        document.getElementById(
            "predictionResult"
        ).innerHTML = `

            <div style="
                font-size:32px;
                font-weight:700;
                margin-bottom:12px;"
                class="${sentimentClass}">
                ${data.label.toUpperCase()}
            </div>

            <div>
                Confidence:
                ${(data.scores[data.label] * 100)
                    .toFixed(2)}%
            </div>

        `;

    } catch (error) {

        console.error(error);

    } finally {

        hideLoading();

    }
}

async function explainSentiment() {

    const text =
        document.getElementById("textInput").value;

    if (!text.trim()) {
        alert("Please enter text.");
        return;
    }

    showLoading();

    try {

        const response =
            await fetch(
                `${API_URL}/explain`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        text
                    })
                }
            );

        const data =
            await response.json();

        let html =
            `<div class="explanation-container">`;

        const maxImpact =
            Math.max(
                ...data.explanation.map(
                    item =>
                        Math.abs(item.impact)
                )
            );

        data.explanation.forEach(item => {

            const width =
                (Math.abs(item.impact)
                    / maxImpact) * 100;

            const barClass =
                item.impact >= 0
                    ? "bar-positive"
                    : "bar-negative";

            html += `

                <div class="explanation-row">

                    <div class="explanation-header">

                        <span class="word-label">
                            ${item.word}
                        </span>

                        <span class="impact-value">
                            ${item.impact.toFixed(3)}
                        </span>

                    </div>

                    <div class="bar-wrapper">

                        <div
                            class="bar ${barClass}"
                            style="width:${width}%">
                        </div>

                    </div>

                </div>

            `;
        });

        html += "</div>";

        document.getElementById(
            "explanationResult"
        ).innerHTML = html;

    } catch (error) {

        console.error(error);

    } finally {

        hideLoading();

    }
}

async function uploadCSV() {

    const fileInput =
        document.getElementById("csvFile");

    const file =
        fileInput.files[0];

    if (!file) {

        alert("Please select a CSV file.");
        return;

    }

    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    const response =
        await fetch(
            `${API_URL}/predict_csv`,
            {
                method: "POST",
                body: formData
            }
        );

    const blob =
        await response.blob();

    const url =
        window.URL.createObjectURL(blob);

    const a =
        document.createElement("a");

    a.href = url;
    a.download =
        "predictions.csv";

    a.click();

    document.getElementById(
        "csvResult"
    ).innerHTML =
        "✅ Predictions downloaded successfully.";
}