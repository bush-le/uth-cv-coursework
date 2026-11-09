# UTH Computer Vision Coursework — Exploratory Data Analysis (EDA) Report

## 1. Executive Summary & Dataset Inventory
- **Total Manifest Records Indexed:** 4000
- **Available Images Verified on Disk:** 3168 (79.20%)
- **Missing Files on Disk:** 832

```
    group  manifest_records  existing_files  missing_files  availability_pct  duplicate_filenames extensions  total_storage_mb  mean_file_size_kb  real_count  fake_count  hard_sample_count
celeba_hq              2000            2000              0             100.0                    0       .jpg            231.83              118.7        2000           0                  0
     ffhq              2000            1168            832              58.4                    0       .png           1495.25             1310.9        2000           0                  0
```

## 2. Image Resolution & Geometric Integrity
```
    group  image_count dominant_resolution  unique_resolutions  mean_width  min_width  max_width  mean_height  min_height  max_height  mean_megapixels  mean_aspect_ratio
celeba_hq         2000           1024x1024                   1      1024.0       1024       1024       1024.0        1024        1024            1.049                1.0
     ffhq         1168           1024x1024                   1      1024.0       1024       1024       1024.0        1024        1024            1.049                1.0
```

## 3. Optical Image Quality & Duplicate Audit
```
  dataset  analyzed_samples  sharpness_mean  sharpness_median  sharpness_p05  brightness_mean  brightness_std  contrast_mean  red_mean  green_mean  blue_mean
celeba_hq               500          291.04            174.13          38.28            112.7           30.78          60.81    132.37      106.63      92.42
     ffhq               500          143.27            101.88          42.65            116.4           28.98          58.16    134.71      110.36      99.47
```

## 4. Cross-Dataset Comparison & Domain Bias Assessment
```
    group  available_images  female_pct  male_pct  young_pct  with_glasses_pct dominant_resolution  mean_aspect_ratio  mean_sharpness  mean_brightness  mean_contrast       rgb_mean_tuple
celeba_hq              2000       77.35     22.65      78.80              3.00           1024x1024                1.0          291.04            112.7          60.81 (132.4, 106.6, 92.4)
     ffhq              1168       50.68     49.32      73.54             21.92           1024x1024                1.0          143.27            116.4          58.16 (134.7, 110.4, 99.5)
```
