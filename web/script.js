/*
  script.js — ImmuCore Frontend Logic
  ======================================

  This script handles all the interactive behavior for the ImmuCore
  health risk prediction form. Right now (Week 1), it does three things:

    1. Form validation — checks that every field has a value within a
       reasonable range before allowing submission.

    2. Placeholder prediction — since the API doesn't exist yet, we
       simulate a result using a simple heuristic based on the input
       values. This lets us test the full UI flow (form → loading →
       results) without needing a backend.

    3. UI state management — showing/hiding the results panel, toggling
       the loading spinner, displaying error messages, etc.

  In Week 3, the placeholder prediction will be replaced with a real
  fetch() call to the FastAPI backend's /predict endpoint.

  No libraries, no frameworks — just plain JavaScript.
*/


// ============================================================
// CONFIGURATION
// Valid ranges for each input field. These come from the Pima
// dataset's actual data ranges plus some reasonable padding.
// We use these for client-side validation.
// ============================================================

const FIELD_CONFIG = {
    pregnancies: {
        min: 0,
        max: 20,
        label: "Pregnancies",
        errorMsg: "Enter a number between 0 and 20"
    },
    glucose: {
        min: 0,
        max: 300,
        label: "Glucose",
        errorMsg: "Enter a value between 0 and 300 mg/dL"
    },
    "blood-pressure": {
        min: 0,
        max: 200,
        label: "Blood Pressure",
        errorMsg: "Enter a value between 0 and 200 mm Hg"
    },
    "skin-thickness": {
        min: 0,
        max: 100,
        label: "Skin Thickness",
        errorMsg: "Enter a value between 0 and 100 mm"
    },
    insulin: {
        min: 0,
        max: 900,
        label: "Insulin",
        errorMsg: "Enter a value between 0 and 900 μU/mL"
    },
    bmi: {
        min: 0,
        max: 70,
        label: "BMI",
        errorMsg: "Enter a value between 0 and 70"
    },
    dpf: {
        min: 0,
        max: 2.5,
        label: "Diabetes Pedigree Function",
        errorMsg: "Enter a value between 0 and 2.5"
    },
    age: {
        min: 1,
        max: 120,
        label: "Age",
        errorMsg: "Enter an age between 1 and 120"
    }
};


// ============================================================
// DOM ELEMENT REFERENCES
// Grab all the elements we'll need to interact with, once,
// at page load time — no need to query them repeatedly.
// ============================================================

const form             = document.getElementById("health-form");
const submitBtn        = document.getElementById("submit-btn");
const resetBtn         = document.getElementById("reset-btn");
const resultsPanel     = document.getElementById("results-panel");
const formSection      = document.querySelector(".form-section");
const severityBadge    = document.getElementById("severity-badge");
const severityCard     = document.getElementById("severity-card");
const confidenceValue  = document.getElementById("confidence-value");
const explanationText  = document.getElementById("explanation-text");
const precautionsList  = document.getElementById("precautions-list");
const newCheckBtn      = document.getElementById("new-check-btn");
const errorBanner      = document.getElementById("error-banner");
const errorMessage     = document.getElementById("error-message");
const errorDismiss     = document.getElementById("error-dismiss");


// ============================================================
// FORM VALIDATION
// Checks each field against its configured min/max range.
// Returns an object: { isValid: bool, errors: [...] }
// ============================================================

function validateForm() {
    /*
      Walk through every field defined in FIELD_CONFIG, grab its
      value from the form, and check:
        1. Is the field filled in? (not empty)
        2. Is the value a valid number? (not NaN)
        3. Is it within the min/max range?

      We collect all errors first, then report them at once —
      rather than stopping at the first error.
    */

    let errors = [];
    let isValid = true;

    // Loop through each field we care about
    for (let fieldId in FIELD_CONFIG) {
        let config = FIELD_CONFIG[fieldId];
        let input = document.getElementById(fieldId);
        let value = input.value.trim();

        // Remove any previous error styling
        input.classList.remove("input-error");

        // Check 1: is the field empty?
        if (value === "") {
            errors.push(config.label + " is required");
            input.classList.add("input-error");
            isValid = false;
            continue;
        }

        // Check 2: is it a valid number?
        let numValue = parseFloat(value);
        if (isNaN(numValue)) {
            errors.push(config.label + " must be a number");
            input.classList.add("input-error");
            isValid = false;
            continue;
        }

        // Check 3: is it within the valid range?
        if (numValue < config.min || numValue > config.max) {
            errors.push(config.errorMsg);
            input.classList.add("input-error");
            isValid = false;
        }
    }

    return { isValid: isValid, errors: errors };
}


// ============================================================
// COLLECT FORM DATA
// Reads all the input values and returns them as a plain object.
// This is the data that will be sent to the API in Week 3.
// ============================================================

function collectFormData() {
    /*
      We return the data with keys that match the dataset's column
      names — this way the API schema (Week 3) won't need any
      name translation.
    */

    return {
        Pregnancies:               parseFloat(document.getElementById("pregnancies").value),
        Glucose:                   parseFloat(document.getElementById("glucose").value),
        BloodPressure:             parseFloat(document.getElementById("blood-pressure").value),
        SkinThickness:             parseFloat(document.getElementById("skin-thickness").value),
        Insulin:                   parseFloat(document.getElementById("insulin").value),
        BMI:                       parseFloat(document.getElementById("bmi").value),
        DiabetesPedigreeFunction:  parseFloat(document.getElementById("dpf").value),
        Age:                       parseFloat(document.getElementById("age").value)
    };
}


// ============================================================
// PLACEHOLDER PREDICTION
// A simple heuristic that mimics what the real model might return.
// This will be replaced with a fetch() call to /predict in Week 3.
// ============================================================

function placeholderPredict(data) {
    /*
      This is NOT a real prediction — it's a rough heuristic so we
      can test the UI flow end-to-end without a backend.

      The heuristic is based on commonly known diabetes risk factors:
        - High glucose is the strongest indicator
        - High BMI increases risk
        - Older age increases risk
        - Family history (DPF) matters

      We compute a simple "risk score" from 0 to 1 and use it to
      determine the severity level and a fake confidence value.
    */

    // Weight each factor by its rough importance
    let riskScore = 0;

    // Glucose: the strongest predictor — values above 140 are concerning
    if (data.Glucose > 180) {
        riskScore += 0.35;
    } else if (data.Glucose > 140) {
        riskScore += 0.25;
    } else if (data.Glucose > 100) {
        riskScore += 0.10;
    }

    // BMI: above 30 is considered obese
    if (data.BMI > 35) {
        riskScore += 0.20;
    } else if (data.BMI > 30) {
        riskScore += 0.12;
    } else if (data.BMI > 25) {
        riskScore += 0.05;
    }

    // Age: risk increases with age
    if (data.Age > 60) {
        riskScore += 0.15;
    } else if (data.Age > 45) {
        riskScore += 0.10;
    } else if (data.Age > 35) {
        riskScore += 0.05;
    }

    // Diabetes Pedigree Function: higher means more family history
    if (data.DiabetesPedigreeFunction > 0.8) {
        riskScore += 0.15;
    } else if (data.DiabetesPedigreeFunction > 0.5) {
        riskScore += 0.08;
    }

    // Blood pressure: above 90 diastolic is high
    if (data.BloodPressure > 90) {
        riskScore += 0.08;
    }

    // Insulin: very high levels can indicate insulin resistance
    if (data.Insulin > 200) {
        riskScore += 0.07;
    }

    // Clamp the score between 0 and 1
    riskScore = Math.min(1.0, Math.max(0.0, riskScore));

    // Determine the severity level based on the score
    let severity, prediction, confidence;

    if (riskScore < 0.25) {
        severity = "low";
        prediction = 0;
        confidence = (0.70 + Math.random() * 0.20).toFixed(2);
    } else if (riskScore < 0.55) {
        severity = "moderate";
        prediction = 1;
        confidence = (0.55 + Math.random() * 0.20).toFixed(2);
    } else {
        severity = "high";
        prediction = 1;
        confidence = (0.75 + Math.random() * 0.20).toFixed(2);
    }

    return {
        prediction: prediction,
        severity: severity,
        confidence: parseFloat(confidence),
        riskScore: riskScore
    };
}


// ============================================================
// GENERATE PLACEHOLDER EXPLANATION
// Fake AI-generated text for the results panel. In Week 3, this
// will come from the LLM via the /explain endpoint.
// ============================================================

function generatePlaceholderExplanation(data, result) {
    /*
      We build a somewhat realistic-looking explanation based on
      the input values and the placeholder prediction result.
      This is just for UI demonstration — the real version will
      come from an LLM.
    */

    let explanations = {
        low: "Based on the health values you provided, the model estimates a "
           + "low probability of diabetes risk. Your glucose levels and BMI are "
           + "within generally healthy ranges. Keep in mind that this is a "
           + "screening tool, not a diagnosis — regular check-ups with your "
           + "healthcare provider are always recommended.",

        moderate: "The model has flagged some of your values as potentially "
                + "indicating an elevated diabetes risk. This does not mean you "
                + "have or will develop diabetes — it simply suggests that some "
                + "of your health markers (such as glucose or BMI) are in ranges "
                + "that warrant attention. Please consult with your healthcare "
                + "provider to discuss these results.",

        high: "Based on the values you entered, the model estimates a higher "
            + "probability of diabetes risk. Several of your health markers — "
            + "particularly glucose levels — fall in ranges that are commonly "
            + "associated with elevated risk. This is not a diagnosis, but it "
            + "is a strong signal to discuss these results with a qualified "
            + "healthcare professional as soon as is practical."
    };

    return explanations[result.severity];
}


function generatePlaceholderPrecautions(result) {
    /*
      Return a list of precautions appropriate to the severity level.
      In Week 3, the LLM will generate these based on the actual
      prediction and input values.
    */

    let precautions = {
        low: [
            "Continue maintaining a balanced diet with whole grains, vegetables, and lean protein",
            "Stay physically active — aim for at least 150 minutes of moderate exercise per week",
            "Schedule regular annual health check-ups to monitor your baseline values",
            "Stay hydrated and maintain a consistent sleep schedule"
        ],
        moderate: [
            "Schedule a follow-up appointment with your healthcare provider to discuss these results",
            "Monitor your blood glucose levels more frequently — consider a home testing kit",
            "Review your diet with a focus on reducing refined sugars and processed carbohydrates",
            "Increase physical activity gradually — even a daily 30-minute walk can make a difference",
            "If you have a family history of diabetes, share this information with your doctor"
        ],
        high: [
            "Consult with a healthcare professional as soon as practical to review these results",
            "Request a formal oral glucose tolerance test (OGTT) or HbA1c test from your doctor",
            "Work with a dietitian to develop a meal plan focused on blood sugar management",
            "Begin a structured exercise routine — discuss safe options with your doctor first",
            "Monitor your blood glucose levels regularly and keep a log for your healthcare provider"
        ]
    };

    return precautions[result.severity];
}


// ============================================================
// DISPLAY RESULTS
// Takes the prediction result and updates the results panel.
// ============================================================

function displayResults(data, result) {
    /*
      This function does three things:
        1. Sets the severity badge (color + text)
        2. Fills in the explanation text
        3. Populates the precautions list

      Then it hides the form and shows the results panel.
    */

    // -- 1. Set the severity badge --
    let badgeText = {
        low: "Low Risk",
        moderate: "Moderate Risk",
        high: "High Risk"
    };

    // Clear any previous severity class, then add the new one
    severityBadge.className = "severity-badge";
    severityBadge.classList.add("severity-" + result.severity);
    severityBadge.textContent = badgeText[result.severity];

    // -- 2. Set the confidence score --
    confidenceValue.textContent = (result.confidence * 100).toFixed(0) + "%";

    // -- 3. Set the explanation text --
    let explanation = generatePlaceholderExplanation(data, result);
    explanationText.textContent = explanation;

    // -- 4. Populate the precautions list --
    let precautions = generatePlaceholderPrecautions(result);
    precautionsList.innerHTML = ""; // Clear any old items

    for (let i = 0; i < precautions.length; i++) {
        let li = document.createElement("li");
        li.textContent = precautions[i];
        precautionsList.appendChild(li);
    }

    // -- 5. Show the results, hide the form --
    formSection.classList.add("hidden");
    resultsPanel.classList.remove("hidden");

    // Scroll to the top of the results so the user sees them immediately
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ============================================================
// ERROR HANDLING
// Show and hide the error banner.
// ============================================================

function showError(message) {
    /*
      Display the error banner with a specific message.
      The banner slides in from the top with a CSS animation.
    */
    errorMessage.textContent = message;
    errorBanner.classList.remove("hidden");

    // Auto-dismiss after 8 seconds so it doesn't sit there forever
    setTimeout(function() {
        hideError();
    }, 8000);
}

function hideError() {
    errorBanner.classList.add("hidden");
}


// ============================================================
// LOADING STATE
// Toggle the submit button between normal and loading states.
// ============================================================

function setLoading(isLoading) {
    /*
      When loading:
        - Disable the submit button so the user can't double-click
        - Show the spinner animation inside the button
        - The button text fades to half opacity
    */
    if (isLoading) {
        submitBtn.classList.add("loading");
        submitBtn.disabled = true;
    } else {
        submitBtn.classList.remove("loading");
        submitBtn.disabled = false;
    }
}


// ============================================================
// EVENT HANDLERS
// Wire up the form submission, reset, and other interactions.
// ============================================================

// -- Form submission --
form.addEventListener("submit", function(event) {
    /*
      When the user clicks "Analyze Risk":
        1. Prevent the default form submission (no page reload)
        2. Validate all fields
        3. If valid, show loading state for a moment (simulating API call)
        4. Run the placeholder prediction
        5. Display the results
    */

    // Don't let the browser submit the form normally
    event.preventDefault();

    // Hide any previous errors
    hideError();

    // Step 1: Validate
    let validation = validateForm();
    if (!validation.isValid) {
        // Show the first error message (keep it simple for the user)
        showError(validation.errors[0]);
        return;
    }

    // Step 2: Collect the form data
    let formData = collectFormData();

    // Step 3: Show loading state
    setLoading(true);

    // Step 4: Simulate an API call with a short delay
    // In Week 3, this setTimeout will be replaced with a real fetch() call.
    setTimeout(function() {
        // Run the placeholder prediction
        let result = placeholderPredict(formData);

        // Turn off the loading spinner
        setLoading(false);

        // Display the results
        displayResults(formData, result);

    }, 1200); // 1.2 second fake delay to show the loading animation
});


// -- "New Check" button (in the results panel) --
newCheckBtn.addEventListener("click", function() {
    /*
      When the user clicks "← New Check", hide the results
      and show the form again so they can enter new values.
    */
    resultsPanel.classList.add("hidden");
    formSection.classList.remove("hidden");

    // Scroll back to the top
    formSection.scrollIntoView({ behavior: "smooth", block: "start" });
});


// -- Form reset --
form.addEventListener("reset", function() {
    /*
      When the user clicks "Clear All", remove any error styling
      from the input fields and hide the error banner.

      The browser handles clearing the input values automatically
      on reset — we just need to clean up our custom UI states.
    */

    // Small delay so the browser's reset happens first
    setTimeout(function() {
        // Remove error styling from all inputs
        let inputs = form.querySelectorAll("input");
        for (let i = 0; i < inputs.length; i++) {
            inputs[i].classList.remove("input-error");
        }

        // Hide any error banner
        hideError();
    }, 10);
});


// -- Dismiss error banner --
errorDismiss.addEventListener("click", function() {
    hideError();
});


// -- Remove error styling when user starts typing in a field --
// This gives immediate feedback that the error is being addressed.
(function() {
    let inputs = form.querySelectorAll("input");
    for (let i = 0; i < inputs.length; i++) {
        inputs[i].addEventListener("input", function() {
            this.classList.remove("input-error");
        });
    }
})();
