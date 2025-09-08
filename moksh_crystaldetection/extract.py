from utils import *


'''
This is the code for extraction and storage of results using the Pipeline class in the utils file,
from the data folder provided by filip called Crystal Images Filip. Can be used for testing and exporting results
for new models.
'''

DATA_DIR = '/Users/pagatok/Projets/Stage/crystal_unresolved'



def crystal_export_results(subdir):
    print("hello")
    resultdir = subdir + '/results'
    pipe = Pipeline(detection_model_path=DETECT_MODEL, segmentation_model_path=SEGMENT_MODEL)
    os.mkdir(resultdir)
    os.mkdir(resultdir +'/segment_res')
    if(subdir.split('/')[-1] == 'crystal images'):
        imglist = list(glob.glob(subdir + '/*.jpg'))
        excel_list = []
        for i, img in enumerate(imglist):
            img_name = img.split('/')[-1]
            pipe.generate_detections(img, 'results')
            out_image, out_list = pipe.display_detections()
            if(len(out_list) > 0):
                size_lookup, sizes = pipe.segment(f'{resultdir}/segment_res')
                excel_list.extend(list(size_lookup.items()))
            plt.imsave(f'{resultdir}/detected_crystals_{img_name}', out_image)
            print(f"{i}. DONE {img}")
        excel_list = [(x.split('/')[-1], y) for (x, y) in excel_list]
        df = pd.DataFrame(excel_list, columns=['filename', 'size'])
        df.to_excel(f'{resultdir}/sizes.xlsx', index=False)

        clean_data = remove_outliers(df['size'].tolist())
        sns.histplot(clean_data, bins=12, kde=True)
        plt.savefig(f'{resultdir}/histogram.jpg')
        print("CRYSTAL SAVED!")

def time_export_results(subdir):
    pipe = Pipeline(detection_model_path=DETECT_MODEL, segmentation_model_path=SEGMENT_MODEL)
    if(subdir.split('/')[-1] == 'time images'):
        print('hi')
        imglist = list(glob.glob(subdir + '/*.jpg'))
        for i, img in enumerate(imglist):
            respath = subdir + '/' + img.split('/')[-1].split('.')[0]
            if not os.path.exists(respath):
                os.mkdir(respath)
            if not os.path.exists(respath + '/segment_res'):
                os.mkdir(respath + '/segment_res')
            img_name = img.split('/')[-1]
            pipe.generate_detections(img, 'results')
            out_image, out_list = pipe.display_detections()
            if(len(out_list) > 0):
                size_lookup, sizes = pipe.segment(f'{respath}/segment_res')
                excel_list = list(size_lookup.items())
            else:
                excel_list = []
            plt.imsave(f'{respath}/detected_crystals_{img_name}', out_image)
            if len(excel_list) > 0:
                excel_list = [(x.split('/')[-1], y) for (x, y) in excel_list]
                df = pd.DataFrame(excel_list, columns=['filename', 'size'])
                df.to_excel(f'{respath}/sizes.xlsx', index=False)
                clean_data = remove_outliers(df['size'].tolist())
                plt.figure(figsize=(8, 8))
                sns.histplot(clean_data, bins=12, kde=True)
                plt.savefig(f'{respath}/histogram.jpg')
                print("SAVED!")
        print(f"{i}. DONE {img}")
                              
if __name__ == '__main__':
    subdir = DATA_DIR
    crystal_paths = []
    # for dir in glob.glob(subdir+'/*/*/*/*/*/time images'):
    #     crystal_paths.append(dir)
    # for dir in glob.glob(subdir + '/*/time images'):
    #     crystal_paths.append(dir)
    for dir in glob.glob(subdir+'/*/*/*/*/*/crystal images'):
        crystal_paths.append(dir)
    for dir in glob.glob(subdir + '/*/crystal images'):
        crystal_paths.append(dir)
    
    for path in crystal_paths:
        print(path)
        # time_export_results(path)
        crystal_export_results(path)