# Data-driven Hallucination of Different Times of Day from a Single Outdoor Photo
### Authors
*Yichang Shih,* 
*Sylvain Paris,* 
*Fredo Durand,* 
*William T. Freeman*

<br>

### Team SeaWeed
*Divya Varshini,*
*Souvik Karfa,*
*Soveet Nayak*

<br>

## System Specification
| Package | Version  |
| ------  | -------- | 
| Python  |  3.8.10  |
| MATLAB   |  R2022b  |
| openCV  |  4.7.0   |
| numpy   |  1.24.3  |
| matplotlib   |  3.7.1  |
| matlabengine   |  9.13.7  |

<br>

## Structure
Following files are included in the folder:

    .
    ├── images                              input image
    ├── src                    
    |   ├── patchMatch.ipynb                code for patchMatch
    |   ├── directColorTransfer.ipynb       code for Color transfer 
    |   ├── mrfMatch.ipynb                  code for mrfMatch and Local affine transfer
    |   └── getLaplacian.m                  helper function for mrfMatch
    |  
    └── README.md                           this file
