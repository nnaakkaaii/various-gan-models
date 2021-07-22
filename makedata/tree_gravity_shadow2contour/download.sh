mkdir -p ./inputs/tree_gravity_shadow2contour
wget -O ./inputs/tree_gravity_shadow2contour.zip https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/wzbzznk8z3-1.zip
wget -O ./inputs/tree_gravity_shadow2contour/filename_train.json "https://drive.google.com/uc?export=download&id=1RonnDX06VK9iXhXAZXXTCLC6RK_4eMJ_"
wget -O ./inputs/tree_gravity_shadow2contour/filename_test.json "https://drive.google.com/uc?export=download&id=1NBPNt49me-BLdVLpX-A14qPx-rTc6MBB"
unzip ./inputs/tree_gravity_shadow2contour.zip -d ./inputs/tree_gravity_shadow2contour/
unzip ./inputs/tree_gravity_shadow2contour/DATASET.zip -d ./inputs/tree_gravity_shadow2contour/
mv ./inputs/tree_gravity_shadow2contour/DATASET/* ./inputs/tree_gravity_shadow2contour/
mv ./inputs/tree_gravity_shadow2contour/'IMAGE PAIRS 273x193px' ./inputs/tree_gravity_shadow2contour/IMAGE_PAIRS_273x193px
mv ./inputs/tree_gravity_shadow2contour/'IMAGE PAIRS 136x96px' ./inputs/tree_gravity_shadow2contour/IMAGE_PAIRS_136x96px
rm ./inputs/tree_gravity_shadow2contour.zip
rm -rf ./inputs/tree_gravity_shadow2contour/Dataset/
rm ./inputs/tree_gravity_shadow2contour/DATASET.zip
rm -rf ./inputs/tree_gravity_shadow2contour/DATASET
rm ./inputs/tree_gravity_shadow2contour/cc-by-4.0.png
rm -rf ./inputs/tree_gravity_shadow2contour/SCRIPTS/
rm -rf ./inputs/tree_gravity_shadow2contour/LEGEND/