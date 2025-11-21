import java.util.Arrays;
import java.util.List;

/**
 * DiabetesRiskPredictor: A PhD Application Showcase for NUS Yong Loo Lin School of Medicine.
 * * This application demonstrates the implementation of a basic, interpretable machine learning
 * model (Logistic Regression) in pure Java. It calculates the probability of a patient developing
 * Type 2 Diabetes and performs model evaluation on a mock 'real-world' dataset.
 * * The emphasis is on transparency, clinical relevance, and core engineering skills, suitable 
 * for a translational research environment.
 */
public class DiabetesRiskPredictor {

    // --- 1. Patient Data Structure & Mock Dataset ---

    /**
     * Record to hold the input features for a patient, including the known clinical outcome (label).
     * @param pregnancies Number of times pregnant (or 0 for males/non-applicable).
     * @param glucose Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
     * @param bmi Body Mass Index (weight in kg/(height in m)^2).
     * @param age Age in years.
     * @param diabetesPedigreeFunction A score reflecting the genetic risk based on family history.
     * @param outcome 1 for Diabetic (Positive), 0 for Non-Diabetic (Negative).
     */
    public record PatientRecord(
        double pregnancies,
        double glucose,
        double bmi,
        double age,
        double diabetesPedigreeFunction,
        int outcome
    ) {
        /**
         * Converts the PatientRecord's features into an array for the model's prediction function.
         * The order MUST match the order of the weights in the LogisticRegressionModel.
         * @return A double array of features.
         */
        public double[] toFeatureArray() {
            return new double[] {
                pregnancies,
                glucose,
                bmi,
                age,
                diabetesPedigreeFunction
            };
        }
    }

    /**
     * A mock 'real-world' dataset (simulating a sample from a clinical cohort).
     * This data is used to evaluate the model's performance.
     */
    private static final PatientRecord[] REAL_WORLD_DATASET = {
        // Patient 1: Low Risk Case
        new PatientRecord(1, 85, 23.5, 25, 0.25, 0),
        // Patient 2: High Risk, Correctly Predicted (True Positive)
        new PatientRecord(4, 150, 33.0, 48, 0.70, 1),
        // Patient 3: Low Risk, Correctly Predicted (True Negative)
        new PatientRecord(0, 99, 21.0, 30, 0.15, 0),
        // Patient 4: High Glucose/BMI, Correctly Predicted (True Positive)
        new PatientRecord(6, 175, 38.5, 55, 0.62, 1),
        // Patient 5: Borderline, Incorrectly Predicted (False Negative - missed case)
        new PatientRecord(2, 140, 27.5, 40, 0.55, 1), 
        // Patient 6: Very Low Risk, Correctly Predicted (True Negative)
        new PatientRecord(0, 80, 20.0, 22, 0.10, 0),
        // Patient 7: High Risk, Correctly Predicted (True Positive)
        new PatientRecord(5, 168, 30.1, 45, 0.88, 1),
        // Patient 8: Misclassified as Positive (False Positive - over-alert)
        new PatientRecord(2, 125, 29.0, 35, 0.35, 0),
        // Patient 9: Moderate Risk, Correctly Predicted (True Positive)
        new PatientRecord(3, 130, 25.5, 50, 0.45, 1),
        // Patient 10: Low Risk, Correctly Predicted (True Negative)
        new PatientRecord(1, 108, 24.1, 28, 0.30, 0)
    };


    // --- 2. Logistic Regression Model Implementation ---

    /**
     * Implements a simple, pre-trained Logistic Regression Model.
     */
    public static class LogisticRegressionModel {
        
        // Hypothetical weights derived from a simulated training process.
        private static final double[] WEIGHTS = {
            0.11,   // Weight for Pregnancies
            0.035,  // Weight for Glucose (High positive weight for high risk)
            0.08,   // Weight for BMI
            0.015,  // Weight for Age
            0.95    // Weight for Diabetes Pedigree Function (High impact genetic factor)
        };
        
        // Hypothetical bias (intercept) term.
        private static final double BIAS = -8.0;

        /**
         * The Sigmoid function (Logistic Function).
         * P(Y=1) = 1 / (1 + e^(-z))
         * @param z The linear combination (dot product of features and weights + bias).
         * @return The predicted probability (risk score) between 0.0 and 1.0.
         */
        private double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        /**
         * Calculates the predicted risk score for a given patient.
         * @param features The array of input patient features.
         * @return The predicted probability of diabetes (0.0 to 1.0).
         */
        public double predict(double[] features) {
            if (features.length != WEIGHTS.length) {
                throw new IllegalArgumentException(
                    "Feature dimension mismatch. Expected " + WEIGHTS.length + 
                    " features, but received " + features.length
                );
            }

            // 1. Calculate the Linear Combination (z = w · x + b)
            double linearCombination = BIAS; // Start with the bias term
            for (int i = 0; i < WEIGHTS.length; i++) {
                linearCombination += WEIGHTS[i] * features[i];
            }

            // 2. Apply the Sigmoid function to get the probability
            return sigmoid(linearCombination);
        }

        /**
         * Evaluates the model's performance on a given dataset and prints key clinical metrics.
         * This demonstrates model validation capability.
         * @param dataset The array of patient records with known outcomes.
         * @param threshold The probability threshold (e.g., 0.50) to classify a prediction as positive.
         */
        public void evaluateDataset(PatientRecord[] dataset, double threshold) {
            int truePositives = 0;  // Correctly predicted diabetic
            int trueNegatives = 0;  // Correctly predicted non-diabetic
            int falsePositives = 0; // Predicted diabetic, but non-diabetic in reality (False Alarm)
            int falseNegatives = 0; // Predicted non-diabetic, but diabetic in reality (Missed Case)

            for (PatientRecord record : dataset) {
                double prediction = predict(record.toFeatureArray());
                int predictedClass = (prediction >= threshold) ? 1 : 0;
                int actualClass = record.outcome();

                if (predictedClass == 1 && actualClass == 1) {
                    truePositives++;
                } else if (predictedClass == 0 && actualClass == 0) {
                    trueNegatives++;
                } else if (predictedClass == 1 && actualClass == 0) {
                    falsePositives++;
                } else { // predictedClass == 0 && actualClass == 1
                    falseNegatives++;
                }
            }

            int total = dataset.length;
            double accuracy = (double) (truePositives + trueNegatives) / total;
            // Sensitivity (Recall): TP / (TP + FN). Ability to correctly identify positive cases. (CRITICAL for screening)
            double sensitivity = (truePositives + falseNegatives) > 0 
                               ? (double) truePositives / (truePositives + falseNegatives) 
                               : 0.0;
            // Specificity: TN / (TN + FP). Ability to correctly identify negative cases.
            double specificity = (trueNegatives + falsePositives) > 0 
                               ? (double) trueNegatives / (trueNegatives + falsePositives) 
                               : 0.0;

            System.out.println("=======================================================================");
            System.out.println("|| Model Evaluation on Mock Real-World Dataset (N=" + total + ") ||");
            System.out.println("=======================================================================");
            System.out.printf("Threshold used: %.2f (%.0f%%)\n\n", threshold, threshold * 100);

            // Display Confusion Matrix components
            System.out.println("--- Confusion Matrix Components ---");
            System.out.printf("  True Positives (TP): %d (Correctly identified as Diabetic)\n", truePositives);
            System.out.printf("  True Negatives (TN): %d (Correctly identified as Non-Diabetic)\n", trueNegatives);
            System.out.printf("  False Positives (FP): %d (False Alarm / Predicted Diabetic, was Negative)\n", falsePositives);
            System.out.printf("  False Negatives (FN): %d (Missed Case / Predicted Negative, was Diabetic)\n", falseNegatives);

            // Display Summary Metrics
            System.out.println("\n--- Performance Metrics ---");
            System.out.printf("  Overall Accuracy (TP+TN/Total): %.2f%%\n", accuracy * 100);
            System.out.printf("  Sensitivity (Recall) (TP/(TP+FN)): %.2f%% (Ability to detect actual cases)\n", sensitivity * 100);
            System.out.printf("  Specificity (TN/(TN+FP)): %.2f%% (Ability to rule out non-cases)\n", specificity * 100);
            System.out.println("-----------------------------------------------------------------------");
        }

        /**
         * Returns the list of features the model uses for interpretation/explanation.
         */
        public List<String> getFeatureNames() {
            return Arrays.asList("Pregnancies", "Glucose", "BMI", "Age", "Diabetes Pedigree Function");
        }
        
        /**
         * Returns the model's weights for interpretability.
         */
        public double[] getWeights() {
            return WEIGHTS;
        }
    }

    // --- 3. Main Application Demonstration ---

    public static void main(String[] args) {
        // Instantiate the model and define the risk threshold
        LogisticRegressionModel model = new LogisticRegressionModel();
        final double RISK_THRESHOLD = 0.50; // 50% risk threshold for intervention

        // --- Demonstrate Model Interpretability ---
        System.out.println("=======================================================================");
        System.out.println("|| Diabetes Risk Predictor (Java ML Showcase) ||");
        System.out.println("|| Target: NUS Yong Loo Lin School of Medicine PhD Program             ||");
        System.out.println("=======================================================================");
        System.out.println("Model: Pure Java Logistic Regression | Intervention Threshold: " + (RISK_THRESHOLD * 100) + "%");
        System.out.println("-----------------------------------------------------------------------");
        
        List<String> featureNames = model.getFeatureNames();
        double[] weights = model.getWeights();
        System.out.println("Model Interpretability: Feature Weights (Impact on Risk):");
        for (int i = 0; i < featureNames.size(); i++) {
            System.out.printf("  %-28s: %+.4f\n", featureNames.get(i), weights[i]);
        }
        System.out.printf("  %-28s: %.4f\n", "Intercept (Bias)", LogisticRegressionModel.BIAS);
        System.out.println("-----------------------------------------------------------------------");


        // --- Single Patient Prediction Demonstration ---
        
        // Case 1: Low Risk (Young, Healthy BMI, Normal Glucose)
        PatientRecord patientA = new PatientRecord(0, 95, 22.5, 30, 0.21, 0);

        // Case 2: High Risk (Obese, High Glucose, Older)
        PatientRecord patientB = new PatientRecord(3, 160, 35.1, 58, 0.75, 1);

        // Case 3: Borderline Risk (Normal Glucose, but high BMI and age)
        PatientRecord patientC = new PatientRecord(1, 105, 31.0, 45, 0.40, 0);

        System.out.println("\n*** Single Patient Risk Stratification ***");
        double riskA = model.predict(patientA.toFeatureArray());
        displayPrediction(patientA, riskA, RISK_THRESHOLD);
        
        double riskB = model.predict(patientB.toFeatureArray());
        displayPrediction(patientB, riskB, RISK_THRESHOLD);

        double riskC = model.predict(patientC.toFeatureArray());
        displayPrediction(patientC, riskC, RISK_THRESHOLD);


        // --- Model Evaluation on Dataset ---
        model.evaluateDataset(REAL_WORLD_DATASET, RISK_THRESHOLD);
        
        System.out.println("=======================================================================");
    }

    /**
     * Helper method to print the prediction results in a formatted way.
     */
    private static void displayPrediction(PatientRecord patient, double riskScore, double threshold) {
        String riskLevel = riskScore >= threshold ? "HIGH" : "LOW";
        String recommendation = riskScore >= threshold 
                                ? "Recommend immediate lifestyle intervention and clinical follow-up." 
                                : "Routine monitoring recommended.";
        
        System.out.println("\n--- Patient Profile ---");
        System.out.printf("  Age: %d, BMI: %.1f, Glucose: %.0f\n", 
            (int)patient.age(), patient.bmi(), patient.glucose());
        System.out.printf("  Genetic Risk (DPF): %.3f\n", patient.diabetesPedigreeFunction());
        
        System.out.println("\n--- Prediction Result ---");
        System.out.printf("  Predicted Diabetes Risk (P(Diabetes)): %.2f%% (%s Risk)\n", 
            riskScore * 100, riskLevel);
        System.out.printf("  Actual Outcome: %s (0=Non-Diabetic, 1=Diabetic)\n", patient.outcome() == 1 ? "1 (Diabetic)" : "0 (Non-Diabetic)");
        System.out.println("  Clinical Recommendation: " + recommendation);
        System.out.println("-----------------------------------------------------------------------");
    }
}
