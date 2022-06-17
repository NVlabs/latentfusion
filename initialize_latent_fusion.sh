LM_FILE_NAME="lm_04_and_09.zip"
LM_FILE2_NAME="lm_format.zip"
mkdir -p /ssd_scratch/cvit/amoghtiwari/latentfusion/
cd /ssd_scratch/cvit/amoghtiwari/latentfusion

mkdir checkpoints
scp amoghtiwari@ada.iiit.ac.in:/share3/amoghtiwari/checkpoints/latentfusion_checkpoints/latentfusion-release.pth checkpoints/

mkdir data
cd data
scp amoghtiwari@ada.iiit.ac.in:/share3/amoghtiwari/data/$LM_FILE_NAME ./
scp amoghtiwari@ada.iiit.ac.in:/share3/amoghtiwari/data/$LM_FILE2_NAME ./
unzip $LM_FILE_NAME
unzip $LM_FILE2_NAME
rm $LM_FILE_NAME
rm $LM_FILE2_NAME

