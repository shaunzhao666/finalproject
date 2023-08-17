import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

name_cla = [
    "Nearest Neighbors",
    "linear SVM",
    "RBF SVM",
    "Neural Network",
    "Naive Bayes",
    "Decision Tree",
    "Logistic Regression", 
    ]


classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel='linear', C=1.0),
    SVC(gamma=0.2, C=50),
    MLPClassifier(hidden_layer_sizes=50, max_iter=1000),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=10),
    LogisticRegression(C=10)
]


hd_df = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/midterm_object/code/heart.csv")
column_name = np.array(hd_df.columns)
variable_name = column_name[0: len(column_name)-1]

ex_df = pd.DataFrame(index=["with heart diseases", "without heart disease"])
target0 = hd_df.where(hd_df["target"] == 0).dropna()
target1 = hd_df.where(hd_df["target"] == 1).dropna()
for i in variable_name:
    ex_df[i] = [target1[i].mean(), target0[i].mean()]

dic = []
de_val = ['age', 'exang', 'thal', 'restecg', 'cp', 'slope', 'ca']
con_val = ['trestbps', 'chol', 'thalach', 'oldpeak']
for i in con_val:
    a = np.array(hd_df.loc[:, i]).reshape(hd_df.shape[0], 1)
    pre = StandardScaler().fit(a)
    dic.append((i, pre))
for j in de_val:
    a = np.array(hd_df.loc[:, j]).reshape(hd_df.shape[0], 1)
    pre = MinMaxScaler().fit(a)
    dic.append((j, pre))
dic = dict(dic)

hd_pre = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/data_afterpreprocessing.csv")
hd_pre = hd_pre.drop(columns="Unnamed: 0")


Y_df = hd_pre["target"].values.copy()
X_df = hd_pre.iloc[:, 0:12].values.copy()

test_size = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
# knn
ele_knn = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/knn_elements.csv")
ele_knn = ele_knn.drop(columns="Unnamed: 0")

# linear svm

c_lsvm = [1, 10, 100]
ele_lsvm = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/lsvm_element.csv")
ele_lsvm = ele_lsvm.drop(columns="Unnamed: 0")
#rbfsvm
gamma_svm = [0.001, 0.02, 0.04, 0.08]
c_svm = [1, 10, 50, 100]
ele_svm = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/svm_element.csv")
ele_svm = ele_svm.drop(columns="Unnamed: 0")

# nn
laysize = [10, 50, 100, 200]
max_iter = [200, 500, 1000]
ele_nn = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/nn_element.csv")
ele_nn = ele_nn.drop(columns="Unnamed: 0")

# nb
ele_nb = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/nb_elements.csv")
ele_nb = ele_nb.drop(columns="Unnamed: 0")

# dt
max_depth = [5, 10, 50, 100]
ele_dt = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/dt_element.csv")
ele_dt = ele_dt.drop(columns="Unnamed: 0")


#lr
C_lr = [10, 100, 200]
ele_lr = pd.read_csv("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/lr_element.csv")
ele_lr = ele_lr.drop(columns="Unnamed: 0")


X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.1, random_state=110)


mod = []
eval = pd.DataFrame(columns=['name', 'tn', 'fp', 'fn', 'tp', 'test_error_rate', 'score'])
i = 0
for name, clf in zip(name_cla, classifiers):
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    mod.append(clf)
    tn1, fp1, fn1, tp1 = metrics.confusion_matrix(Y_test, prediction).ravel()
    new = {'name': name, 'tn':tn1, 'fp': fp1, 'fn': fn1, 'tp': tp1, 'test_error_rate': (fn1 + fp1)/(tn1+fp1 + fn1 + tp1), 'score': clf.score(X_test, Y_test)}
    eval.append(new, ignore_index = True)

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
st.sidebar.markdown("author: Shuangyu Zhao")
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
    body = st.container()
    with head:
        st.markdown("# DATASET ANALYSIS")
        
    with body:
        ev, cor, pair, scatter = st.tabs(("expected value", "correlation", "distribution", "3d scatter plot"))

        with ev:
            st.markdown("<h2 style='text-align: center; color: gray;'> Expected Value of Different Attributes </h2>", unsafe_allow_html=True)
            st.table(ex_df.style.set_properties(**{'background-color': 'yellow'}, subset=["cp", "thalach", "exang", "oldpeak", "ca"])\
                .set_table_styles(styles).format(precision=2).set_properties(**{'color': 'blue'}, subset=['sex', 'exang', 'thal', 'restecg', 'age', 'cp', 'slope', 'ca', 'fbs']))
            st.markdown("* The attributes in blue are discrete, ones in black are continuous")
            st.markdown("* The attibutes highlighted by yellow have great differences between the expected value of people with heart disease and without heart disease.")
            st.markdown("* The table shows people with heart disease have higher chest pain, higher maximum heart rate achieved, lower number of major vessels, ST depression and \
                    most of them don't have exercise induced angina.")

        with cor: 
            st.markdown("## correlation between features")
        
            choice = st.selectbox('choose one correlation plot', ('all features', 'customise inputting variables'))
            if choice == 'all features':
                fig = plt.figure(figsize=(16, 16))
                corr = sns.heatmap(hd_df.corr(), annot=True).set_title("correlation map among all features", fontsize=30)
                st.pyplot(fig) 
            else: 
                choice2 = st.multiselect("customise inputting variables", column_name, default=column_name[0])
        
                df = hd_df.loc[:, choice2]
                st.markdown("### you select: {}".format(", ".join(choice2)))

                fig = plt.figure()
                corr = sns.heatmap(df.corr(), annot=True).set_title("correlation map among chosen features", fontsize=30)
                st.pyplot(fig)
            st.write("In the attributes with continuous value, old peak and thalach have high absolute \
                    correlations, age, trestbps and chol's correlations are relatively low. ")
            st.write("In the attibutes with discrete value, fbs only have two values, but the\
                     correlation with target is only -0.041, so I can confirm that fbs cannot help to predict the\
                    heart disease. Sex, cp, thal, exang, slope and ca have high absolute correlations \
                    with target, and restecg's absolute correlation is relatviely low.")

        with pair:
            st.markdown("## pairplot between features")
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
    
            st.markdown("According to the plots above, the distributions of target=0 and target=1 have a little difference in age, exang, oldpeak, thalch and so on, who have high correlations with target.\
                But, their 2d scatterplots show great overlaps between these two results.")
        
    
    
        with scatter:
            st.markdown('## 3D scatterplot among variables')
            col1, col2, col3 = st.columns(3)
            with col1:
                x = st.selectbox("please choose x", variable_name)
            with col2:
                y = st.selectbox("please choose y", variable_name)
            with col3:
                z = st.selectbox("please choose z", variable_name)
            fig3 = px.scatter_3d(hd_df, x=x, y=y, z=z, color='target')
            st.plotly_chart(fig3)

            st.markdown("According to the 3d scatter plot, if we set x=exang, y=oldpeak and z=thalch, whose correlations with target are high, we can find the overlap situation decreases a lot compared with 2d scatter plot.\
                We can find great distributions' differences in 3d scatter plot.")


elif select_page == 'MACHINE LEARNING':
    st.markdown("# MACHINE LEARNING")
    pre = st.container()
    evaluation = st.container()
    ml = st.container()
    summ = st.container()
    with pre:
        st.markdown('### Preprocessing')
        st.image("https://raw.githubusercontent.com/shaunzhao666/finalproject/dataset/preprocess.png", width=900)
    with evaluation:
        st.markdown('### Evaluation for models')
        col1, col2 = st.columns(2)
        with col1: 
            st.markdown('#### Precision: ')
            st.markdown('##### The percentage of true predicted in all predicted positive')
            st.latex(r'''precision = \frac{TP}{TP + FP}''')
            st.markdown('#### Recall: ')
            st.markdown('##### The percetage of true predicted in all actual positive')
            st.latex(r'''recall = \frac{TP}{TP + FN}''')
        with col2: 
            st.markdown('#### F1-score: ')
            st.markdown('##### Showing the precision and recall trade-off ')
            st.latex(r'''f1-score = \frac{2}{\frac{1}{precision}+\frac{1}{recall}} = \frac{2\times{TP}}{2\times{TP} + FP + FN }''')
            st.markdown('#### Accuracy:')
            st.markdown('##### The percentage of true predicted')
            st.latex(r'''accuracy = \frac{TP +TN}{TP+TN+FP+FN}''')
        
    with ml: 
        knn, lsvm, svm, nn , nb, dt, lr = st.tabs(name_cla)
        with knn:
            st.header("K Nearest Neighbor")
            st.markdown("#### test size's influence")

            fig6 = plt.figure(figsize=(8, 8))
            plt.plot(test_size, ele_knn.loc[:, "accuracy"], label='accuracy')
            plt.plot(test_size, ele_knn.loc[:, "precision"], label='precision')
            plt.plot(test_size, ele_knn.loc[:, "recall"], label="recall")
            plt.plot(test_size, ele_knn.loc[:, "f1_score"], label='f1-score')
            plt.xlabel("test_size")
            plt.ylabel("percentage")
            plt.grid(alpha=.5)
            plt.legend()
            st.pyplot(fig6)
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation </h2>", unsafe_allow_html=True)
            st.table(ele_knn[ele_knn["accuracy"] == ele_knn["accuracy"].max()])

            ele_knn1 = ele_knn.where(ele_knn["test_size"]==0.1).dropna()
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation when test size=0.1 </h2>", unsafe_allow_html=True)
            st.table(ele_knn1[ele_knn1["accuracy"] == ele_knn1["accuracy"].max()])
            

            # col1, col2, col3, col4 = st.columns(4)
            # col1.metric("True Positive", tp[0])
            # col2.metric("True Negative", tn[0])
            # col3.metric("False Positive", fp[0])
            # col4.metric("False Negative", fn[0])
            # st.markdown(f"test error rate : {test_error_rate[0]: .3f}")
            # st.markdown(f"the score of model: {score[0]: .3f}")

        with lsvm:
            st.header("Linear SVM")
            st.markdown("#### influence of test_size and C")
            accuracy = np.array(ele_lsvm.iloc[:, 2])
            precision = np.array(ele_lsvm.iloc[:, 3])
            recall = np.array(ele_lsvm.iloc[:, 4])
            f1 = np.array(ele_lsvm.iloc[:, 5])
            fig6 = go.Figure(data = [go.Mesh3d(x=np.array(ele_lsvm.iloc[:, 0]), 
                                    y=np.array(ele_lsvm.iloc[:, 1]), 
                                    z=accuracy,
                                    opacity=.5, 
                                    color='yellow',
                                    name="accuracy")])
                                    
            fig6.add_trace(go.Mesh3d(x=np.array(ele_lsvm.iloc[:, 0]), 
                                    y=np.array(ele_lsvm.iloc[:, 1]), 
                                    z=precision, 
                                    opacity=0.5,
                                    color='pink',
                                    name="precision"))

            fig6.add_trace(go.Mesh3d(x=np.array(ele_lsvm.iloc[:, 0]), 
                                    y=np.array(ele_lsvm.iloc[:, 1]), 
                                    z=recall, 
                                    opacity=0.5,
                                    color='blue',
                                    name="recall"))
            fig6.add_trace(go.Mesh3d(x=np.array(ele_lsvm.iloc[:, 0]), 
                                    y=np.array(ele_lsvm.iloc[:, 1]), 
                                    z=f1, 
                                    opacity=0.5,
                                    color='green',
                                    name="f1_score"))

            fig6.update_layout(scene = dict(
                    xaxis_title='test_size',
                    yaxis_title='C',
                    zaxis_title='percentage'))
            fig6.update_traces(showlegend=True)
            st.plotly_chart(fig6)
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation </h2>", unsafe_allow_html=True)
            st.table(ele_lsvm[ele_lsvm["accuracy"] == ele_lsvm["accuracy"].max()])

            ele_lsvm1 = ele_lsvm.where(ele_lsvm["test_size"]==0.1).dropna()
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation when test size=0.1 </h2>", unsafe_allow_html=True)
            st.table(ele_lsvm1[ele_lsvm1["accuracy"] == ele_lsvm1["accuracy"].max()])
           
        with svm:
            st.header("RBF SVM")
            st.markdown("#### influence of test_size, C and gamma")
            fig = make_subplots(rows=2, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                           [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}]], subplot_titles=("testsize=0.05", "testsize=0.1", "testsize=0.3", "testsize=0.5", "testsize=0.7", "testsize=0.9"))
            x = ele_svm.iloc[:16, 1]
            y = ele_svm.iloc[:16, 2]
            row = [1, 1, 1, 2, 2, 2]
            col = [1, 2, 3, 1, 2, 3]
            n=0
            for i in range(6):
                accuracy = ele_svm.iloc[n:n+16, 3]
                precision = ele_svm.iloc[n:n+16, 4]
                recall = ele_svm.iloc[n:n+16, 5]
                f1 = ele_svm.iloc[n:n+16, 6]

                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=accuracy,
                                    opacity=.5, 
                                    color='yellow',
                                    name="accuracy"), 
                                    row = row[i], col=col[i])
                                    
                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=precision, 
                                    opacity=0.5,
                                    color='pink',
                                    name="precision"), 
                                    row=row[i], col=col[i])

                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=recall, 
                                    opacity=0.5,
                                    color='blue',
                                    name="recall"), 
                                    row=row[i], col=col[i])

                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=f1, 
                                    opacity=0.5,
                                    color='green',
                                    name="f1_score"), 
                                    row=row[i], col=col[i])
                n += 16
            

            fig.update_layout(scene = dict(
                        xaxis_title='gamma',
                        yaxis_title='C',
                        zaxis_title='percentage'),
                        scene2 = dict(
                        xaxis_title='gamma',
                        yaxis_title='C',
                        zaxis_title='percentage'),
                        scene3 = dict(
                        xaxis_title='gamma',
                        yaxis_title='C',
                        zaxis_title='percentage'),
                        scene4 = dict(
                        xaxis_title='gamma',
                        yaxis_title='C',
                        zaxis_title='percentage'),
                        scene5 = dict(
                        xaxis_title='gamma',
                        yaxis_title='C',
                        zaxis_title='percentage'),
                        scene6 = dict(
                        xaxis_title='gamma',
                        yaxis_title='C',
                        zaxis_title='percentage'),)
               

            st.plotly_chart(fig)
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation </h2>", unsafe_allow_html=True)
            st.table(ele_svm[ele_svm["accuracy"] == ele_svm["accuracy"].max()])

            ele_svm1 = ele_svm.where(ele_svm["test_size"]==0.1).dropna()
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation when test size=0.1 </h2>", unsafe_allow_html=True)
            st.table(ele_svm1[ele_svm1["accuracy"] == ele_svm1["accuracy"].max()])
        

        with nn:
            st.header("Neural Network")
            st.markdown("#### influence of test_size, hidden_layers and max_iteration")
            fig = make_subplots(rows=2, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                           [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}]], subplot_titles=("testsize=0.05", "testsize=0.1", "testsize=0.3", "testsize=0.5", "testsize=0.7", "testsize=0.9"))
            x = ele_nn.iloc[:12, 1]
            y = ele_nn.iloc[:12, 2]
            row = [1, 1, 1, 2, 2, 2]
            col = [1, 2, 3, 1, 2, 3]
            n=0
            for i in range(6):
                accuracy = ele_nn.iloc[n:n+12, 3]
                precision = ele_nn.iloc[n:n+12, 4]
                recall = ele_nn.iloc[n:n+12, 5]
                f1 = ele_nn.iloc[n:n+12, 6]

                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=accuracy,
                                    opacity=.5, 
                                    color='yellow',
                                    name="accuracy"), 
                                    row = row[i], col=col[i])
                                    
                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=precision, 
                                    opacity=0.5,
                                    color='pink',
                                    name="precision"), 
                                    row=row[i], col=col[i])

                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=recall, 
                                    opacity=0.5,
                                    color='blue',
                                    name="recall"), 
                                    row=row[i], col=col[i])

                fig.append_trace(go.Mesh3d(x=x, 
                                    y=y, 
                                    z=f1, 
                                    opacity=0.5,
                                    color='green',
                                    name="f1_score"), 
                                    row=row[i], col=col[i])
                n += 12
            

            fig.update_layout(scene = dict(
                        xaxis_title='layer_size',
                        yaxis_title='max_iter',
                        zaxis_title='percentage'),
                        scene2 = dict(
                        xaxis_title='layer_size',
                        yaxis_title='max_iter',
                        zaxis_title='percentage'),
                        scene3 = dict(
                        xaxis_title='layer_size',
                        yaxis_title='max_iter',
                        zaxis_title='percentage'),
                        scene4 = dict(
                        xaxis_title='layer_size',
                        yaxis_title='max_iter',
                        zaxis_title='percentage'),
                        scene5 = dict(
                        xaxis_title='layer_size',
                        yaxis_title='max_iter',
                        zaxis_title='percentage'),
                        scene6 = dict(
                        xaxis_title='layer_size',
                        yaxis_title='max_iter',
                        zaxis_title='percentage'),)
               

            st.plotly_chart(fig)
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation </h2>", unsafe_allow_html=True)
            st.table(ele_nn[ele_nn["accuracy"] == ele_nn["accuracy"].max()])

            ele_nn1 = ele_nn.where(ele_nn["test_size"]==0.1).dropna()
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation when test size=0.1 </h2>", unsafe_allow_html=True)
            st.table(ele_nn1[ele_nn1["accuracy"] == ele_nn1["accuracy"].max()])


        with nb:
            st.header("Naive bayes")
            st.markdown("#### test size's influence")
            fig6 = plt.figure(figsize=(8, 8))
            plt.plot(test_size, ele_nb.loc[:, "accuracy"], label='accuracy')
            plt.plot(test_size, ele_nb.loc[:, "precision"], label='precision')
            plt.plot(test_size, ele_nb.loc[:, "recall"], label="recall")
            plt.plot(test_size, ele_nb.loc[:, "f1_score"], label='f1-score')
            plt.xlabel("test_size")
            plt.ylabel("percentage")
            plt.grid(alpha=.5)
            plt.legend()
            st.pyplot(fig6)
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation </h2>", unsafe_allow_html=True)
            st.table(ele_nb[ele_nb["accuracy"] == ele_nb["accuracy"].max()])

            ele_nb1 = ele_nb.where(ele_nb["test_size"]==0.1).dropna()
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation when test size=0.1 </h2>", unsafe_allow_html=True)
            st.table(ele_nb1[ele_nb1["accuracy"] == ele_nb1["accuracy"].max()])

        with dt:
            st.header("Decision Tree")
            st.markdown("#### influence of test_size and max_depth")
            accuracy = np.array(ele_dt.iloc[:, 2])
            precision = np.array(ele_dt.iloc[:, 3])
            recall = np.array(ele_dt.iloc[:, 4])
            f1 = np.array(ele_dt.iloc[:, 5])
            fig6 = go.Figure(data = [go.Mesh3d(x=np.array(ele_dt.iloc[:, 0]), 
                                    y=np.array(ele_dt.iloc[:, 1]), 
                                    z=accuracy,
                                    opacity=.5, 
                                    color='yellow',
                                    name="accuracy")])
                                    
            fig6.add_trace(go.Mesh3d(x=np.array(ele_dt.iloc[:, 0]), 
                                    y=np.array(ele_dt.iloc[:, 1]), 
                                    z=precision, 
                                    opacity=0.5,
                                    color='pink',
                                    name="precision"))

            fig6.add_trace(go.Mesh3d(x=np.array(ele_dt.iloc[:, 0]), 
                                    y=np.array(ele_dt.iloc[:, 1]), 
                                    z=recall, 
                                    opacity=0.5,
                                    color='blue',
                                    name="recall"))
            fig6.add_trace(go.Mesh3d(x=np.array(ele_dt.iloc[:, 0]), 
                                    y=np.array(ele_dt.iloc[:, 1]), 
                                    z=f1, 
                                    opacity=0.5,
                                    color='green',
                                    name="f1_score"))

            fig6.update_layout(scene = dict(
                    xaxis_title='test_size',
                    yaxis_title='max_depth',
                    zaxis_title='percentage'))
            fig6.update_traces(showlegend=True)
            st.plotly_chart(fig6)
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation </h2>", unsafe_allow_html=True)
            st.table(ele_dt[ele_dt["accuracy"] == ele_dt["accuracy"].max()])

            ele_dt1 = ele_dt.where(ele_dt["test_size"]==0.1).dropna()
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation when test size=0.1 </h2>", unsafe_allow_html=True)
            st.table(ele_dt1[ele_dt1["accuracy"] == ele_dt1["accuracy"].max()])
        
        with lr: 
            st.header("Logistic Regression")
            st.markdown("#### influence of test_size and C")
            accuracy = np.array(ele_lr.iloc[:, 2])
            precision = np.array(ele_lr.iloc[:, 3])
            recall = np.array(ele_lr.iloc[:, 4])
            f1 = np.array(ele_lr.iloc[:, 5])
            fig6 = go.Figure(data = [go.Mesh3d(x=np.array(ele_lr.iloc[:, 0]), 
                                    y=np.array(ele_lr.iloc[:, 1]), 
                                    z=accuracy,
                                    opacity=.5, 
                                    color='yellow',
                                    name="accuracy")])
                                    
            fig6.add_trace(go.Mesh3d(x=np.array(ele_lr.iloc[:, 0]), 
                                    y=np.array(ele_lr.iloc[:, 1]), 
                                    z=precision, 
                                    opacity=0.5,
                                    color='pink',
                                    name="precision"))

            fig6.add_trace(go.Mesh3d(x=np.array(ele_lr.iloc[:, 0]), 
                                    y=np.array(ele_lr.iloc[:, 1]), 
                                    z=recall, 
                                    opacity=0.5,
                                    color='blue',
                                    name="recall"))
            fig6.add_trace(go.Mesh3d(x=np.array(ele_lr.iloc[:, 0]), 
                                    y=np.array(ele_lr.iloc[:, 1]), 
                                    z=f1, 
                                    opacity=0.5,
                                    color='green',
                                    name="f1_score"))

            fig6.update_layout(scene = dict(
                    xaxis_title='test_size',
                    yaxis_title='C',
                    zaxis_title='percentage'))
            fig6.update_traces(showlegend=True)
            st.plotly_chart(fig6)
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation </h2>", unsafe_allow_html=True)
            st.table(ele_lr[ele_lr["accuracy"] == ele_lr["accuracy"].max()])

            ele_lr1 = ele_lr.where(ele_lr["test_size"]==0.1).dropna()
            st.markdown("<h2 style='text-align: center; color: gray;'> Best situation when test size=0.1 </h2>", unsafe_allow_html=True)
            st.table(ele_lr1[ele_lr1["accuracy"] == ele_lr1["accuracy"].max()])


    with summ:
        st.markdown("## Summary")
        st.markdown(" Even though the best situation's test_sizes are different among different models, \
            all models have good performances when test_size=0.1. Therefore, test_size is chosen as 0.1 for comparisons among models.")
        st.markdown("<h2 style='text-align: center; color: gray;'> The parameters chosen for global comparison </h2>", unsafe_allow_html=True)
        st.table(pd.DataFrame(data = {"knn": [" ", "0.1"], \
                                        "linear svm": ["C=1.0", "0.1"], \
                                            "RBF SVM": ["gamma=0.2 \n C=50", "0.1"], \
                                                "Neural Network": ["hidden_layer=50 \n max_iteration=1000", "0.1"], \
                                                    "Naive Bayes": [" ", "0.1"], \
                                                        "Decision Tree": ["max_depth=10", "0.1"], \
                                                            "Logistic Regression": ["C=10", "0.1"]}, index=["hyperparameter", "test_size"]))
        st.markdown("<h2 style='text-align: center; color: gray;'> The results </h2>", unsafe_allow_html=True)
        st.table(eval)
        fig4 = plt.figure()
        plt.barh(eval["name"], eval["score"])
        plt.xlabel("Accuracy")
        plt.ylabel("Classification Method")
        plt.title("The comparision of accuracies among difference classification")
        st.pyplot(fig4)

        fig5, ax = plt.subplots()
        ax.bar(eval["name"], eval["tp"], color='r', label="True Positive")
        ax.bar(eval["name"], eval["tn"], color='b', label="True Negative", bottom=eval["tp"])
        ax.bar(eval["name"], eval["fp"], color='g', label="False Pegative", bottom=eval["tp"]+eval["tn"])
        ax.bar(eval["name"], eval["fn"], color='k', label="False Negative", bottom=eval["tp"]+eval["tn"]+eval["fp"])
        ax.set_title("The result of test")
        ax.set_xlabel("Classification Method")
        ax.set_ylabel("num")
        plt.xticks(rotation=90)
        ax.legend(bbox_to_anchor= (1.04, 1), loc="upper left")
        st.pyplot(fig5)

        st.markdown("Decision Tree, Neural Network, and RBF SVM are best in the heart disease prediction.")
        st.markdown("In the next chapter, Decision Tree is chosen for your heart disease prediciton.")

if select_page == 'PREDICTION':
    st.markdown("# PREDICTION")
    st.markdown("The prediction below is based on the Decision Tree.")
    st.subheader("user defined prediction")
    age = st.slider("your age", value=50, min_value=20, max_value=100)
    choose_sex = st.radio("your sex", ("female", "male"))
    if choose_sex == "female":
        sex = 0
    elif choose_sex == "male":
        sex = 1

    cp = st.radio("your chest pain type", (0, 1, 2, 3))

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
        list = np.array([ dic["age"].transform([[age]]), sex, dic["cp"].transform([[cp]]), dic["trestbps"].transform([[trestbps]]) \
            , dic["chol"].transform([[chol]]), dic["restecg"].transform([[restecg]]), \
                dic["thalach"].transform([[thalach]]), dic["exang"].transform([[exang]]), dic["oldpeak"].transform([[oldpeak]]), \
                    dic["slope"].transform([[slope]]), dic["ca"].transform([[ca]]), dic["thal"].transform([[thal]])]).reshape(1, -1)
        re = mod[5].predict(pd.DataFrame(list))
        if re == 1: 
            st.snow()
            st.warning("be careful, you may have heart disease", icon="😷")
            st.warning("futher check, please", icon="🧐")
        if re == 0:
            st.balloons()
            st.warning("whoo... relief, you do not have heart disease", icon="🥳")

st.sidebar.markdown("### reference")
st.sidebar.markdown("[1] J. P. Li, A. U. Haq, S. U. Din, J. Khan, A. Khan and A. Saboor, \
    'Heart Disease Identification Method Using Machine Learning Classification in E-Healthcare,' \
        in IEEE Access, vol. 8, pp. 107562-107582, 2020, doi: 10.1109/ACCESS.2020.3001149.")

    

