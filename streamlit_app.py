import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

name_cla = [
    "Nearest Neighbors",
    "linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Neural Network",
    "Naive Bayes",
    "Decision Tree"
    ]


classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel='linear', C=0.025),
    SVC(gamma=0.001),
    GaussianProcessClassifier(1.0*RBF(4.0)),
    MLPClassifier(max_iter=1000),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=3),
]


hd_df = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/midterm_object/code/heart.csv")
column_name = np.array(hd_df.columns)
variable_name = column_name[0: len(column_name)-1]

ex_df = pd.DataFrame(index=["with heart diseases", "without heart disease"])
target0 = hd_df.where(hd_df["target"] == 0).dropna()
target1 = hd_df.where(hd_df["target"] == 1).dropna()
for i in variable_name:
    ex_df[i] = [target1[i].mean(), target0[i].mean()]

Y_df = hd_df["target"].values.copy()
X_df = hd_df.iloc[:, 0:13].values.copy()
X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=1)

mod = []
score = []
tn = []
tp = []
fn = []
fp = []
test_error_rate = []
for name, clf in zip(name_cla, classifiers):
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    mod.append(clf)
    tn1, fp1, fn1, tp1 = metrics.confusion_matrix(Y_test, prediction).ravel()
    tn.append(tn1)
    fp.append(fp1)
    fn.append(fn1)
    tp.append(tp1)
    test_error_rate.append((fn1 + fp1)/(tn1+fp1 + fn1 + tp1))
    score.append(clf.score(X_test, Y_test))

th_props = [
  ('font-size', '20px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7ffff')
  ]
                               
td_props = [
  ('font-size', '15px'),
  ('font-weight', 'bold'),
  ]
                               
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]






# app
st.sidebar.title("NAVIGATION")
select_page = st.sidebar.radio("SELECT A PAGE", ('INTRODUCTION', 'ANALYSIS', 'MACHINE LEARNING', 'PREDICTION'))
if select_page == 'INTRODUCTION':
    st.markdown("# HEART DISEASE PREDICTION")
    st.image("https://images.ctfassets.net/yixw23k2v6vo/6BezXYKnMqcG4LSEcWyXlt/b490656e99f34bc18999f3563470eae6/iStock-1156928054.jpg", width=900)
    st.text("image link: https://images.ctfassets.net/yixw23k2v6vo/6BezXYKnMqcG4LSEcWyXlt/b490656e99f34bc18999f3563470eae6/iStock-1156928054.jpg")
    st.markdown("<h2 style='text-align: center; color: gray;'>Heart Disease Dataset </h2>", unsafe_allow_html=True)
    st.table(hd_df.head().style.set_table_styles(styles).format(precision=2).set_properties(**{'text-align': 'left'}))
    st.markdown(""" meaning of variables: 
- age: in years
- sex: 1-male; 0-female
- cp: chest pain type(0, 1, 2, 3)
- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholestoral in mg/dl
- fbs: fasting blood sugar > 120 mg/dl (true:1; false:0)
- restech: resting electrocardiographic results (0,1,2)
- thalach: maximum heart rate achieved
- exang: exercise induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
- ca: number of major vessels (0-3) colored by flourosopy
- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
- target: have heart disease = 1; no = 0""")
    st.text("The heart disease dataset comes from: ")
    st.markdown("https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")


elif select_page == 'ANALYSIS':
    head = st.container()
    corrplot = st.container()
    pairplot = st.container()
    dis = st.container()
    with head:
        st.markdown("# DATASET ANALYSIS")
        st.markdown("<h2 style='text-align: center; color: gray;'> Expected Value of Different Attributes </h2>", unsafe_allow_html=True)
        st.table(ex_df.style.set_properties(**{'background-color': 'yellow'}, subset=["cp", "thalach", "exang", "oldpeak", "ca"])\
            .set_table_styles(styles).format(precision=2).set_properties(**{'color': 'blue'}, subset=['sex', 'exang', 'thal', 'restecg', 'age', 'cp', 'slope', 'ca', 'fbs']))
        st.markdown("* The attributes in blue are discrete, ones in black are continuous")
        st.markdown("* The attibutes highlighted by yellow have great differences between the expected value of people with heart disease and without heart disease.")
        st.markdown("* The table shows people with heart disease have higher chest pain, higher maximum heart rate achieved, lower number of major vessels, ST depression and \
                    most of them don't have exercise induced angina.")

    with corrplot:
        st.markdown("## 1. correlation between features")
            
        choice = st.selectbox('choose one correlation plot', ('all features', 'customise inputting variables'))
        if choice == 'all features':
            fig = plt.figure(figsize=(16, 16))
            corr = sns.heatmap(hd_df.corr(), annot=True).set_title("correlation map among all features", fontsize=30)
            st.pyplot(fig) 
        else: 
            choice2 = st.multiselect("customise inputting variables", column_name, default=column_name[0])
        
            df = hd_df.loc[:, choice2]
            st.markdown("### you select: {}".format(", ".join(choice2)))

            fig = plt.figure(figsize=(16, 16))
            corr = sns.heatmap(df.corr(), annot=True).set_title("correlation map among chosen features", fontsize=30)
            st.pyplot(fig)
        st.write("In the attributes with continuous value, old peak and thalach have high absolute \
                    correlations, age, trestbps and chol's correlations are relatively low. ")
        st.write("In the attibutes with discrete value, fbs only have two values, but the\
                     correlation with target is only -0.041, so I can confirm that fbs cannot help to predict the\
                    heart disease. Sex, cp, thal, exang, slope and ca have high absolute correlations \
                    with target, and restecg's absolute correlation is relatviely low.")

    with pairplot:
        st.markdown("## 2. pairplot between features")
        col1, col2 = st.columns([1, 3])
        with col1: 
            st.markdown("Choose the variables")
            var = np.zeros(len(variable_name) + 1)
            var[-1] = 1
            for i in range(len(variable_name)):
                var[i] = st.checkbox(variable_name[i])

        with col2:
            if sum(var) == 1:
                st.error("please choose the variables", icon='😁')
            else: 
                snsplot_df = hd_df.loc[:, column_name[var==1]]
                fig2 = sns.pairplot(snsplot_df, hue="target")
                st.pyplot(fig2)
                st.markdown("for targe, 0 means not having heartdisease, and 1 means having heart disease")
    
    
        
    
    
    with dis:
        st.markdown('## 3. 3D scatterplot between variables')
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.selectbox("please choose x", variable_name)
        with col2:
            y = st.selectbox("please choose y", variable_name)
        with col3:
            z = st.selectbox("please choose z", variable_name)
        fig3 = px.scatter_3d(hd_df, x=x, y=y, z=z, color='target')
        st.plotly_chart(fig3)


elif select_page == 'MACHINE LEARNING':
    st.markdown("# MACHINE LEARNING")
    ml = st.container()
    summ = st.container()
    with ml: 
        knn, lsvm, svm, gp, nn , nb, dt = st.tabs(name_cla)
        with knn:
            st.header("K Nearest Neighbor")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp[0])
            col2.metric("True Negative", tn[0])
            col3.metric("False Positive", fp[0])
            col4.metric("False Negative", fn[0])
            st.markdown(f"test error rate : {test_error_rate[0]: .3f}")
            st.markdown(f"the score of model: {score[0]: .3f}")

        with lsvm:
            st.header("Linear SVM")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp[1])
            col2.metric("True Negative", tn[1])
            col3.metric("False Positive", fp[1])
            col4.metric("False Negative", fn[1])
            st.markdown(f"test error rate : {test_error_rate[1]: .3f}")
            st.markdown(f"the score of model: {score[1]: .3f}")

        with svm:
            st.header("RBF SVM")
        
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp[2])
            col2.metric("True Negative", tn[2])
            col3.metric("False Positive", fp[2])
            col4.metric("False Negative", fn[2])
            st.markdown(f"test error rate : {test_error_rate[2]: .3f}")
            st.markdown(f"the score of model: {score[2]: .3f}")
        
        with gp:
            st.header("Gaussian Process")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp[3])
            col2.metric("True Negative", tn[3])
            col3.metric("False Positive", fp[3])
            col4.metric("False Negative", fn[3])
            st.markdown(f"test error rate : {test_error_rate[3]: .3f}")
            st.markdown(f"the score of model: {score[3]: .3f}")

        with nn:
            st.header("Neural Network")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp[4])
            col2.metric("True Negative", tn[4])
            col3.metric("False Positive", fp[4])
            col4.metric("False Negative", fn[4])
            st.markdown(f"test error rate : {test_error_rate[4]: .3f}")
            st.markdown(f"the score of model: {score[4]: .3f}")

        with nb:
            st.header("Naive bayes")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp[5])
            col2.metric("True Negative", tn[5])
            col3.metric("False Positive", fp[5])
            col4.metric("False Negative", fn[5])
            st.markdown(f"test error rate : {test_error_rate[5]: .3f}")
            st.markdown(f"the score of model: {score[5]: .3f}")

        with dt:
            st.header("Decision Tree")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Positive", tp[6])
            col2.metric("True Negative", tn[6])
            col3.metric("False Positive", fp[6])
            col4.metric("False Negative", fn[6])
            st.markdown(f"test error rate : {test_error_rate[6]: .3f}")
            st.markdown(f"the score of model: {score[6]: .3f}")

    with summ:
        st.markdown("## Summary")
        # df_Score = pd.DataFrame(score, index=name_cla)
        fig4 = plt.figure()
        plt.barh(name_cla, score)
        plt.xlabel("Accuracy")
        plt.ylabel("Classification Method")
        plt.title("The comparision of accuracies among difference classification")
        st.pyplot(fig4)

        fig5 = plt.figure()
        plt.barh(name_cla, tp, color='r', label="True Positive")
        plt.barh(name_cla, tn, color='b', label="True Negative")
        plt.barh(name_cla, fp, color='g', label="False Negative")
        plt.barh(name_cla, fn, color='k', label="False Negative")
        plt.title("The result of test")
        plt.xlabel("num")
        plt.ylabel("Classification Method")
        plt.legend(bbox_to_anchor= (1.04, 1), loc="upper left")
        st.pyplot(fig5)

if select_page == 'PREDICTION':
    st.markdown("# PREDICTION")
    st.markdown("The best classification is Gaussian Process.")
    st.markdown("The prediction below is based on the Gaussian Process")
    st.subheader("user defined prediction")
    age = st.slider("your age", value=50, min_value=20, max_value=100)
    choose_sex = st.radio("your sex", ("female", "male"))
    if choose_sex == "female":
        sex = 0
    elif choose_sex == "male":
        sex = 1
    cp = st.radio("your chest pain type", (0, 1, 2, 3))
    fbs_choose = st.radio("your fast bool sugar>120mg/dl", ("yes", "no"))
    if fbs_choose == "yes":
        fbs = 1
    elif fbs_choose =="no":
        fbs = 0

    trestbps = st.slider("your resting blood presure(trestbps)", value=100, min_value=90, max_value=200)
    chol = st.slider("your serum cholestoral(chol)", value=300, min_value=100, max_value=500)
    restecg = st.radio("your resting electrocardiographic results(restecg)", (0, 1, 2))
    thalach = st.slider("your maximum heart rate achieved(thalach)", value=135, min_value=70, max_value=200)
    choose_exang = st.radio("your exercise induced angina(exang)", ("yes", "no"))
    if choose_exang == "yes":
        exang = 1
    elif choose_exang == "no":
        exang = 0
    oldpeak = st.slider("your ST depression induced by exercise relative to rest(oldpeak)", value=3.25, min_value=0.0, max_value=6.5)
    slope = st.radio("your slope of the peak exercise ST segment(slope)", (0, 1, 2))
    ca = st.radio("your number of major vessels colored by flourosopy(ca)", (0, 1, 2, 3, 4))
    choose_thal = st.radio("your thal", ("normal", "fixed defect", "reversable defect"))
    if choose_thal == "normal":
        thal = 0
    elif choose_thal == "fixed defect":
        thal = 1
    elif choose_thal == "reversable defect":
        thal = 2
    if st.button("predict"):
        list = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        re = mod[3].predict(pd.DataFrame(list))
        if re == 1: 
            st.snow()
            st.warning("be careful, you may have heart disease", icon="😷")
            st.warning("futher check, please", icon="🧐")
        if re == 0:
            st.balloons()
            st.warning("whoo... relief, you do not have heart disease", icon="🥳")


    

