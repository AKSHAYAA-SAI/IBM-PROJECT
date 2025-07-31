# IBM-PROJECT
 Developed a smart chatbot using IBM Watsonx Assistant and deployed it on IBM Cloud. Integrated with WhatsApp via Twilio for real-time user interaction. Gained hands-on experience in cloud services, AI workflows, and chatbot automation.
<embed src="C:\Users\gobik\Downloads\IBM PROJECT REPORT.pdf" type="application/pdf" width="100%" height="600px" />
# PROBLEM STATEMENT
 In today's digital landscape, the widespread accessibility of information and 
generative AI tools has made it increasingly challenging to ensure the originality of 
academic and professional content. Traditional plagiarism detection systems often 
fall short when it comes to identifying intelligently paraphrased or AI-generated 
submissions that closely mimic authentic work. This project proposes the 
development of a machine learning-based content verification system capable of 
classifying submitted texts into categories such as Original, Human-Paraphrased, AI
Generated, and Verbatim Plagiarism. By analyzing both the structure and semantic 
similarity between the original and submitted content, the system will be able to 
detect subtle signs of AI involvement or sophisticated content manipulation. 
Furthermore, beyond academic applications, the system aims to address the growing 
concern of fake or AI-generated legal documents by flagging contracts, affidavits, and 
other critical texts that may not originate from qualified professionals. This solution 
not only supports educators in upholding academic integrity but also assists legal 
professionals in maintaining the authenticity and reliability of formal documentation.
# PROPOSED SOLUTION
 # 1 . Data Collection
 •Use the PAN Corpusfor 
structured plagiarism 
examples.

 •Generate AI-written content
 using tools like ChatGPT.
 
 •Include human-labeled 
documents(academic or 
legal) with clear 
classifications

 # 2. Data Preprocessing
 •Clean and normalizetext 
(remove stopwords, 
punctuation).

 •Encode labels(Original, AI
Generated, etc.) for model 
training.

 •Extract featuresusing TF-IDF 
or embeddings for 
meaningful input.

# 3. Machine Learning Algorithm 
(IBM Watsonx.ai AutoAI)
 •Leverage AutoAIto auto
select and train the best 
models.

 •Perform feature engineering 
and hyperparameter tuning
 automatically.
 
 •Choose the optimal model 
based on F1-score and 
precision

# 4. Deployment
 •Deploy the model via 
Watsonx.ai as an API 
endpoint.

 •Integrate with a simple UIfor 
users to upload and check 
documents.

 •Enable real-time predictions
 for academic or legal input 
detection.

 # 5. Evaluation
 •Use accuracy, precision, 
recall, and F1-scoreto 
assess performance.

 •Visualize results using a 
confusion matrixto spot 
misclassifications.

 •Ensure transparency and 
fairnessusing Watson 
OpenScale.

 # 6.Results.
 •Achieved high classification 
accuracy(typically 90%+ 
using AutoAI) in 
distinguishing between 
original, AI-generated, and 
plagiarized content.

 •Successfully detected AI
generated legal and 
academic documentswith 
strong precision, reducing 
the risk of fake submissions.
# SYSTEM APPROACH
 ibm-watson-machine-learning
 •To connect your local  Python environment 
with Watsonx.ai and manage AutoAI
 training and deployments.
 pandas
 •For reading, cleaning, and preparing your 
dataset (e.g., CSV files) before uploading to 
Watsonx.
 scikit-learn (optional)
 •Useful for preprocessing steps like TF-IDF, 
encoding, or local evaluation before or 
after using AutoAI.
 Libraries 
Required to 
Build the Model 
(Watsonx.ai)
# ALGORITHM & DEPLOYMENT
  Algorithm Selection
 1. IBM Watsonx.ai AutoAI selects the best algorithm (e.g., Logistic Regression, Random Forest, or Gradient Boosting) based on the 
dataset characteristics.
 2. The selection is automated but optimized to handle text similarity classification tasks using model evaluation metrics like F1
score.
  Data Input
 1. Input features include original text, submitted text, and derived features such as TF-IDF or semantic embeddings.
 2. Additional metadata like source type (ChatGPT, PAN, etc.) and instructor feedback can improve contextual detection.
  Training Process
 1. The dataset is split into training and test sets within Watsonx.ai, and AutoAI applies cross-validation to avoid overfitting.
 2. AutoAI performs automated hyperparameter tuning to optimize the selected model's performance.
  Prediction Process
 1. The trained model compares new submission pairs (original vs. submitted text) to predict labels like AI_Generated or Plagiarized.
 2. Real-time document input via an app or API is processed instantly, providing a prediction along with confidence scores.
# RESULT
<img width="1887" height="909" alt="Screenshot 2025-07-23 194638" src="https://github.com/user-attachments/assets/f2fa9a1d-c5de-4c06-85b2-6aadf228481f" />

<img width="1915" height="911" alt="Screenshot 2025-07-24 213108" src="https://github.com/user-attachments/assets/d0f0d864-a289-4d6f-9d7b-6e271a0dfbd7" />

<img width="1915" height="906" alt="Screenshot 2025-07-24 213047" src="https://github.com/user-attachments/assets/4cbc3f03-8414-47e3-aa55-e48cb4ab4324" />

<img width="1915" height="909" alt="Screenshot 2025-07-24 212856" src="https://github.com/user-attachments/assets/99242721-cb6d-486f-8d5a-f6db242f548f" />

<img width="1916" height="904" alt="Screenshot 2025-07-24 212834" src="https://github.com/user-attachments/assets/a0de6ba2-e5fa-4b48-a907-a12271dfb2d6" />

<img width="1915" height="909" alt="Screenshot 2025-07-24 212524" src="https://github.com/user-attachments/assets/c2d5abc4-185f-4cc0-bc64-4165dcd80abc" />

<img width="1904" height="907" alt="Screenshot 2025-07-23 194730" src="https://github.com/user-attachments/assets/a64f47ee-16a3-4d8b-9947-897cf86e4a95" />

# CONCLUSION
  This project presents an AI-powered plagiarism and document authenticity detection system 
designed to address the rising challenges posed by generative AI and paraphrasing tools. By 
leveraging IBM Watsonx.ai's AutoAI capabilities, the solution automates model selection, training, 
and evaluation, making it scalable and efficient. The model accurately distinguishes between 
original, paraphrased, AI-generated, and plagiarized content across academic and legal domains. 
With its ability to deliver real-time, explainable predictions, this system not only supports academic 
integrity but also plays a crucial role in verifying the authenticity of professional documents —
 making it a practical and impactful tool for educational institutions and legal professionals alike

# FUTURE SCOPE
  The future scope of this project offers significant potential for expansion and impact. One major 
enhancement would be integrating the model with Learning Management Systems (LMS) like 
Moodle or Google Classroom to automate real-time plagiarism checks during submissions. 
Support for multiple languages could broaden the system's applicability across global academic 
and legal institutions. By incorporating advanced semantic models such as BERT or RoBERTa, the 
system can better understand nuanced content and improve detection accuracy. Additionally, 
embedding explainable AI tools like IBM Watson OpenScale would enhance transparency by 
clarifying why certain documents are flagged. Beyond academics and legal sectors, the solution 
can be adapted to identify AI-generated content in resumes, reports, financial documents, and 
more. Providing real-time, formative feedback could also support ethical writing and content 
authenticity across various fields

# REFERENCES
 •PAN Plagiarism Corpus– Webis Group, https://pan.webis.de/clef.html
 (Dataset for plagiarism detection research)
 •IBM Watsonx.ai Documentation– IBM, https://www.ibm.com/cloud/watsonx-ai
 (Used for building, training, and deploying AI models with AutoAI)
 •Potthast, M., et al. (2010). Evaluating plagiarism detection – PAN'10 Lab overview. CLEF.
 (Research work outlining methods in plagiarism detection)
 •Scikit-learn Documentation– https://scikit-learn.org
 (Library used for data preprocessing and ML experimentation)
 •ChatGPT by OpenAI– https://openai.com/chatgpt
 (Used to generate AI-written text samples for training and testing)
 •Alzahrani, S. M., Salim, N., & Abraham, A. (2012). Understanding plagiarism linguistic patterns, 
textual features, and detection methods. IEEE Transactions on Systems, Man, and Cybernetics
