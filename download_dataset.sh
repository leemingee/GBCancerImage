# Download an example slide and tumor mask
# Note: these are hosted on Google Cloud Storage.
# The remainder are in a Google Drive folder, linked above.
# mkdir data
cd data
FILE=$1
echo "$FILE"

slide_path_091='tumor_'${FILE}'.tif'
tumor_mask_path_091='tumor_'${FILE}'_mask.tif'

slide_url_091='https://storage.googleapis.com/applied-dl/%s' % slide_path_091
mask_url_091='https://storage.googleapis.com/applied-dl/%s' % tumor_mask_path_091

# Download the whole slide image
if not os.path.exists(slide_path_$FILE):
  !curl -O $slide_url_$FILE

# Download the tumor mask
if not os.path.exists(tumor_mask_path_$FILE):
  !curl -O $mask_url_$FILE