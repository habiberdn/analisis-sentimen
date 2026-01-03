import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from lib.scrape import download
from component.sidebar import render_sidebar

@st.dialog("Combine CSV")
def combine():
    firstCSV = st.file_uploader("First CSV", key="first_csv")
    secondCSV = st.file_uploader("Second CSV", key="second_csv")

    if st.button("Combine & Download"):
        if firstCSV is None or secondCSV is None:
            st.warning("Upload kedua file CSV terlebih dahulu")
            return
        csv_data = download(firstCSV, secondCSV)
        st.success("CSV berhasil digabung")
        st.download_button(
            label="Download Combined CSV",
            data=csv_data,
            file_name="combined.csv",
            mime="text/csv",
            use_container_width=True
        )

# Auth Configuration
@st.cache_resource
def load_config():
    with open("config.yaml") as file:
        return yaml.load(file, Loader=SafeLoader)

config = load_config()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# Login
authenticator.login(max_concurrent_users=2)
auth_status = st.session_state.get("authentication_status")

# Handle Authentication
if auth_status is False:
    st.error("Username atau password salah")

elif auth_status is None:
    st.info("Silakan login untuk melanjutkan")
    st.markdown(
        """
        <style>
        /* Hide the sidebar itself */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Hide the button to expand/collapse the sidebar */
        [data-testid="collapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

elif auth_status is True:
     # Pages
    pg = st.navigation(
         [
             st.Page("pages/home.py", title="Home"),
             st.Page("pages/scrape_page.py", title="Scrapping"),
             st.Page("pages/processing_page.py", title="Preprocess"),
             st.Page("pages/analyze_page.py", title="Classification"),
         ],
         position="sidebar"
    )
    pg.run()
    with st.sidebar:
        render_sidebar(authenticator)

# Styling
st.markdown("""
    <style>
        .stTextInput input, .stNumberInput input {
            background-color: #f0f8ff;
            color: #333333;
        }
        ul[data-testid="stSidebarNavItems"] {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        a {
            padding: 4px;
        }
        button[aria-label="Close"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)
