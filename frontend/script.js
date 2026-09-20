async function analyzeMeeting() {

    const transcript = document.getElementById("transcript").value;
    const button = document.getElementById("analyzeBtn");
    const loading = document.getElementById("loading");
    const result = document.getElementById("result");

    if (!transcript.trim()) {
        alert("Please enter a transcript.");
        return;
    }

    button.disabled = true;
    loading.classList.remove("hidden");
    result.classList.add("hidden");

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/api/analyze",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    transcript: transcript
                })
            }
        );

        if (!response.ok) {
            throw new Error("Failed to analyze meeting");
        }

        const data = await response.json();

        displayResults(data);

    } catch (error) {

        alert("Something went wrong. Make sure Flask is running.");

        console.error(error);

    } finally {

        button.disabled = false;
        loading.classList.add("hidden");

    }
}


function displayResults(data) {

    document.getElementById("result").classList.remove("hidden");

    document.getElementById("summary").textContent =
        data.summary || "No summary available.";

    displayActions(data.action_items);
    displayList("suggestions", data.suggestions);
    displayList("decisions", data.decisions);
    displayList("information", data.information);
}


function displayActions(actions) {

    const container = document.getElementById("actions");

    if (!actions || actions.length === 0) {
        container.innerHTML = '<p class="empty">No action items found.</p>';
        return;
    }

    container.innerHTML = actions.map(item => `
        <div class="action">
            <strong>${item.person}</strong>

            <div>
                ${item.action}
            </div>

            <small>
                Deadline: ${item.deadline || "Not specified"}
            </small>
        </div>
    `).join("");
}


function displayList(id, items) {

    const container = document.getElementById(id);

    if (!items || items.length === 0) {
        container.innerHTML =
            '<p class="empty">None found.</p>';
        return;
    }

    container.innerHTML = items.map(item => `
        <div class="item">${item}</div>
    `).join("");
}