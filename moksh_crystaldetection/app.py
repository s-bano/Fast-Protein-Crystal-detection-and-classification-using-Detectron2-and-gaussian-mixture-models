
from utils import *

'''
This file runs the streamlit interface to visualize results from the model.
run:- streamlit run app.py to access the web interface
'''

def main():

    pipe = Pipeline(detection_model_path=DETECT_MODEL, segmentation_model_path=SEGMENT_MODEL)

    st.title('Crystal Detection')
    uploaded_file = st.file_uploader('Upload Image: ', type=['jpg', 'png', 'tif'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img.save('input.jpg')

        pipe.generate_detections('input.jpg', 'results')
        out_image, out_list = pipe.display_detections()
        plt.imsave('results/detected_crystals.jpg', out_image)
        st.title('Detections')
        st.image(out_image)
        if(len(out_list) > 0):
            size_lookup, sizes = pipe.segment('results')
            clean_data = np.array(remove_outliers(list(size_lookup.values())))
            st.title('Filtered Histogram of Sizes')
            fig, ax = plt.subplots()
            sns.histplot(clean_data, bins=10, kde=True, ax=ax)
            ax.set_title("Histogram of Cleaned Data")
            st.pyplot(fig)
            st.text(f'Mean: {np.mean(clean_data)}')
            st.text(f'Standard DeviationL {np.std(clean_data)}')
            
        num_cols = st.slider("Images per row", 1, 5, 3)
        cols = st.columns(num_cols)
        st.title("Extracted Images:")
        #display_list = [x for (x, size) in size_lookup.items() if size in clean_data]
        #st.image(display_list, width=200)
        display_list = list(map(lambda x : cv2.resize(x, (200, 200)),out_list))
        for idx, obj in enumerate(display_list):
            col = cols[idx%num_cols]
            with col:
                st.image(obj, caption=f"crystal {idx+1}", use_container_width=True)
        #st.image(list(map(lambda x : cv2.resize(x, (200, 200)),out_list)), )
                              
if __name__ == '__main__':  
    main()

            
 