import streamlit as st
import pandas as pd

st.title("Real Estate Buyer Segmentation Dashboard")

clients = pd.read_csv("clients.csv")

st.write("Dataset Preview")

st.dataframe(clients.head())
st.sidebar.header("Filters")

country = st.sidebar.selectbox(
    "Country",
    ["All"] + list(clients['country'].unique())
)

if country != "All":
    clients = clients[
        clients['country'] == country
    ]
region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(clients['region'].unique())
)
client_type = st.sidebar.selectbox(
    "Client Type",
    ["All"] + sorted(clients['client type'].unique())
)
purpose = st.sidebar.selectbox(
    "Purpose",
    ["All"] + sorted(clients['acquisition purpose'].unique())
)
filtered = clients.copy()

if country != "All":
    filtered = filtered[filtered['country']==country]

if region != "All":
    filtered = filtered[filtered['region']==region]

if client_type != "All":
    filtered = filtered[filtered['client type']==client_type]

if purpose != "All":
    filtered = filtered[filtered['acquisition purpose']==purpose]


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Create age
clients['date of birth'] = pd.to_datetime(
    clients['date of birth'],
    errors='coerce'
)

clients['age'] = 2026 - clients['date of birth'].dt.year

# Encode categorical fields
le = LabelEncoder()

clients['client type'] = le.fit_transform(clients['client type'])
clients['acquisition purpose'] = le.fit_transform(clients['acquisition purpose'])
clients['loan applied'] = le.fit_transform(clients['loan applied'])

# Features for clustering
X = clients[['age', 'satisfaction score']].fillna(0)

# Create clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clients['Cluster'] = kmeans.fit_predict(X)
cluster_count = clients['Cluster'].value_counts()
import plotly.express as px
st.subheader("📊Cluster Distribution")
cluster_count = clients['Cluster'].value_counts()

fig = px.bar(
    x=cluster_count.index,
    y=cluster_count.values,
    labels={'x':'Cluster','y':'Count'},
    title='Buyer Segments'
)

st.plotly_chart(fig)
st.subheader("🌍Geographical Analysis")
country_count = clients['country'].value_counts()

fig = px.pie(
    values=country_count.values,
    names=country_count.index,
    title="Buyers by Country"
)

st.plotly_chart(fig)

region_count = clients['region'].value_counts()

fig = px.pie(
    values=region_count.values,
    names=region_count.index,
    title='Regional Distribution'
)

st.plotly_chart(fig)

fig = px.histogram(
    clients,
    x='country',
    color='Cluster',
    barmode='group',
    title='Country-wise Buyer Segments'
)

st.plotly_chart(fig)

st.subheader("💰Investor Behavior Dashboard")

fig = px.histogram(
    clients,
    x='acquisition purpose',
    color='Cluster',
    title='Investment Purpose by Cluster'
)

st.plotly_chart(fig)

fig = px.histogram(
    clients,
    x='loan applied',
    color='Cluster',
    title='Loan Usage by Cluster'
)

st.plotly_chart(fig)

fig = px.histogram(
    clients,
    x='referral channel',
    color='Cluster',
    title='Referral Channel Performance'
)

st.plotly_chart(fig)

fig = px.box(
    clients,
    x='Cluster',
    y='satisfaction score',
    title='Satisfaction by Cluster'
)

st.plotly_chart(fig)

st.subheader("🧠 Segment Insights")

summary = clients.groupby(
    'Cluster'
).agg({
    'age':'mean',
    'satisfaction score':'mean'
})

st.dataframe(summary)
clients.to_csv(
    "clustered clients.csv",
    index=False
)
clients = pd.read_csv(
    "clustered clients.csv"
)


st.dataframe(
    clients.groupby('Cluster').describe()
)

segment_names = {
    0:'Global Investors',
    1:'First-Time Buyers',
    2:'Corporate Buyers',
    3:'Luxury Investors'
}
st.write(segment_names)

st.subheader("💼Business Interpretation")

st.markdown("""
### Cluster 0 – Global Investors
- Investment focused
- Less loan dependency

### Cluster 1 – First-Time Buyers
- Younger customers
- High loan dependency

### Cluster 2 – Corporate Buyers
- Business purchases
- Multiple properties

### Cluster 3 – Luxury Investors
- High satisfaction
- Premium investments
""")