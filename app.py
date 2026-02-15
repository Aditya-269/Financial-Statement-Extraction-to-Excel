#!/usr/bin/env python3
"""
Research Portal - Main Application
Upload financial PDFs and extract to Excel automatically
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import time

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_financial_extractor import UniversalFinancialExtractor
from gemini_enhanced_extractor import GeminiEnhancedExtractor


def make_columns_unique(df):
    """Make DataFrame column names unique by adding suffixes to duplicates"""
    if df.empty:
        return df
    
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


def prepare_df_for_display(df):
    """Prepare DataFrame for Streamlit display by ensuring Arrow compatibility"""
    if df.empty:
        return df
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure all columns have consistent types
    for col in df.columns:
        # Skip if all values are None/NaN
        if df[col].isna().all():
            continue
            
        # Check if column contains numeric values
        non_null_values = df[col].dropna()
        if len(non_null_values) == 0:
            continue
        
        # Try to detect if this should be a numeric column
        numeric_count = 0
        for val in non_null_values:
            if isinstance(val, (int, float)):
                numeric_count += 1
            elif isinstance(val, str):
                # Check if string looks like a number
                clean_val = val.replace(',', '').replace('(', '').replace(')', '').replace('-', '').strip()
                if clean_val.replace('.', '').isdigit():
                    numeric_count += 1
        
        # If most values are numeric, convert column to string for safe display
        # This prevents Arrow serialization errors with mixed types
        if numeric_count > len(non_null_values) * 0.5:
            # Convert to string to ensure consistency
            df[col] = df[col].apply(lambda x: '' if pd.isna(x) or x == '' or x is None else str(x))
    
    return df


# Page config
st.set_page_config(
    page_title="Financial Research Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stDownloadButton button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">📊 Financial Research Portal</p>', unsafe_allow_html=True)
    st.markdown("### Extract Financial Statements from PDF to Excel")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/financial-analytics.png", width=150)
        st.markdown("## About")
        st.info("""
        This portal extracts financial statement line items from PDF documents 
        and converts them to structured Excel files for analysis.
        """)
        
        st.markdown("## ⚙️ Settings")
        
        # LLM Enhancement toggle
        use_llm = st.checkbox(
            "Enable AI Enhancement (Gemini)",
            value=False,
            help="Use Google Gemini AI to improve extraction accuracy"
        )
        
        if use_llm:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=os.getenv('GEMINI_API_KEY', ''),
                help="Enter your Google Gemini API key"
            )
            if gemini_key:
                os.environ['GEMINI_API_KEY'] = gemini_key
                st.success("✓ API key set")
            else:
                st.warning("⚠️ Enter API key to enable AI enhancement")
        
        st.markdown("---")
        
        st.markdown("## Supported Documents")
        st.markdown("""
        - ✅ Annual Reports
        - ✅ Quarterly Statements
        - ✅ Income Statements
        - ✅ P&L Statements
        """)
        
        st.markdown("## Features")
        st.success("""
        - Multi-year extraction
        - Automatic normalization
        - Excel export
        - Missing data handling
        """)
        
        st.markdown("---")
        st.caption("Powered by Tesseract OCR & OpenCV")
    
    # Main content (removed tabs - single page interface)
    with st.container():
        st.header("Upload Financial Statement PDF")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload an annual report or financial statement PDF"
            )
        
        with col2:
            st.markdown("### Quick Start")
            st.markdown("""
            1. Upload PDF
            2. Click Extract
            3. Download Excel
            """)
        
        if uploaded_file:
            st.success(f"✓ Uploaded: **{uploaded_file.name}** ({uploaded_file.size/1024:.1f} KB)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                extract_button = st.button("🚀 Extract Data", type="primary", use_container_width=True)
            
            if extract_button:
                with st.spinner("🔄 Processing PDF... Please wait..."):
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status = st.empty()
                        
                        # Extract
                        status.text("📄 Detecting financial pages...")
                        progress_bar.progress(20)
                        
                        # Choose extractor based on LLM setting
                        if use_llm and os.getenv('GEMINI_API_KEY'):
                            extractor = GeminiEnhancedExtractor(tmp_path, use_llm=True)
                            status.text("🤖 Using AI-enhanced extraction...")
                        else:
                            extractor = UniversalFinancialExtractor(tmp_path)
                            status.text("📄 Using standard extraction...")
                        
                        progress_bar.progress(40)
                        
                        status.text("🔍 Extracting table data...")
                        progress_bar.progress(60)
                        
                        # Create output path
                        output_dir = Path("output")
                        output_dir.mkdir(exist_ok=True)
                        output_path = output_dir / f"extracted_{uploaded_file.name.replace('.pdf', '.xlsx')}"
                        
                        status.text("📊 Processing and creating Excel...")
                        progress_bar.progress(80)
                        
                        df = extractor.extract(str(output_path))
                        
                        # Handle duplicate column names for display
                        df = make_columns_unique(df)
                        df = prepare_df_for_display(df)
                        
                        if df.empty:
                            st.error("❌ No financial data found in the PDF")
                            
                            st.warning("### Possible Reasons:")
                            st.markdown("""
                            1. **PDF doesn't contain a financial statement** (income statement/P&L)
                            2. **PDF is password protected** or corrupted
                            3. **Line items have non-standard names**
                            4. **PDF is image-based** with poor OCR quality
                            """)
                            
                            st.info("### 💡 Try This:")
                            st.markdown("""
                            - Upload an **annual report** or **quarterly financial statement**
                            - Ensure the PDF contains line items like:
                              - Revenue from operations
                              - Employee expenses
                              - Profit before tax
                            - Try the **Tata Motors sample** first to verify system is working
                            """)
                            
                            # Offer to test with sample
                            if st.button("🧪 Test with Tata Motors Sample PDF"):
                                st.info("Testing with sample file...")
                                with st.spinner("Processing sample..."):
                                    if use_llm and os.getenv('GEMINI_API_KEY'):
                                        sample_extractor = GeminiEnhancedExtractor(
                                            "Tata Motors Quarterly Financial Statements.pdf",
                                            use_llm=True
                                        )
                                    else:
                                        sample_extractor = UniversalFinancialExtractor(
                                            "Tata Motors Quarterly Financial Statements.pdf"
                                        )
                                    sample_df = sample_extractor.extract("output/sample_test.xlsx")
                                    
                                    # Handle duplicate column names and prepare for display
                                    sample_df = make_columns_unique(sample_df)
                                    sample_df = prepare_df_for_display(sample_df)
                                    
                                    if not sample_df.empty:
                                        st.success("✅ Sample extraction works! Your uploaded PDF may have a different format.")
                                        st.dataframe(sample_df, width='stretch')
                                    else:
                                        st.error("❌ Even sample failed. Please check Tesseract installation.")
                        else:
                            progress_bar.progress(100)
                            status.text("✅ Extraction complete!")
                            
                            time.sleep(0.5)
                            
                            # Success message
                            st.success("🎉 **Extraction completed successfully!**")
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Line Items", len(df))
                            with col2:
                                st.metric("Columns", len(df.columns))
                            with col3:
                                years = [col for col in df.columns if col != 'Particulars']
                                st.metric("Years", len(years))
                            with col4:
                                st.metric("Status", "✓ Ready")
                            
                            # Preview
                            st.markdown("### 📊 Data Preview")
                            st.dataframe(df, width='stretch', height=400)
                            
                            # Download button
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Excel File",
                                    data=f,
                                    file_name=output_path.name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            # Show years detected
                            if years:
                                st.info(f"📅 **Years detected:** {', '.join(years)}")
                    
                    except Exception as e:
                        st.error(f"❌ **Error:** {str(e)}")
                        with st.expander("Show error details"):
                            st.exception(e)
                    
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        else:
            st.info("👆 **Please upload a PDF file to begin**")
            
            # Test sample button
            if os.path.exists("Tata Motors Quarterly Financial Statements.pdf"):
                if st.button("🧪 Test with Sample PDF", help="Test with Tata Motors PDF", use_container_width=True):
                    st.info("Running test extraction...")
                    with st.spinner("Processing..."):
                        try:
                            if use_llm and os.getenv('GEMINI_API_KEY'):
                                test_extractor = GeminiEnhancedExtractor(
                                    "Tata Motors Quarterly Financial Statements.pdf",
                                    use_llm=True
                                )
                            else:
                                test_extractor = UniversalFinancialExtractor(
                                    "Tata Motors Quarterly Financial Statements.pdf"
                                )
                            df = test_extractor.extract("output/sample_demo.xlsx")
                            
                            if not df.empty:
                                st.success("✅ Test successful! System is working correctly.")
                                st.dataframe(df, use_container_width=True)
                                
                                with open("output/sample_demo.xlsx", 'rb') as f:
                                    st.download_button(
                                        "⬇️ Download Sample Output",
                                        data=f,
                                        file_name="tata_motors_sample.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            else:
                                st.error("Test failed. Check Tesseract installation.")
                        except Exception as e:
                            st.error(f"Error: {e}")
    

      
        
        # Show actual extracted file if exists
        if os.path.exists("output/final_extraction.xlsx"):
            st.markdown("---")
            st.markdown("### 📊 Actual Extraction (Tata Motors)")
            df_actual = pd.read_excel("output/final_extraction.xlsx")
            st.dataframe(df_actual, use_container_width=True)
            
            with open("output/final_extraction.xlsx", 'rb') as f:
                st.download_button(
                    label="⬇️ Download Sample Output",
                    data=f,
                    file_name="tata_motors_sample.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Financial Statement Extractor | Built with Streamlit & Tesseract OCR"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
