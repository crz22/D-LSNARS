#include "io2.h"

void getAllFiles(string path, vector<string>& files,string fileType) 
{
	long long hFile = 0;
	// file information
	struct _finddata_t fileinfo;  
 
	string p;
 
	if ((hFile = _findfirst(p.assign(path).append("\\*" + fileType).c_str(), &fileinfo)) != -1) {
		do {
			// save file path
			files.push_back(p.assign(path).append("\\").append(fileinfo.name));
 
		   } while (_findnext(hFile, &fileinfo) == 0);  //Find the next one, return 0 successfully, otherwise -1

		_findclose(hFile);
	}
}

bool getVolume1(V3DPluginCallback2 &callback, char *imagePath, Image4DSimple &VolumeImage, V3DLONG xb, V3DLONG xe,
                  V3DLONG yb, V3DLONG ye, V3DLONG zb, V3DLONG ze) {
    Image4DSimple originImage = *callback.loadImage(imagePath);
    V3DLONG totalBytes = originImage.getTotalBytes();
    //cout<<"originImage.getTotalBytes() : "<<originImage.getTotalBytes()<<endl;
    V3DLONG originSize[4];
    originSize[0] = originImage.getXDim();
    originSize[1] = originImage.getYDim();
    originSize[2] = originImage.getZDim();
    originSize[3] = originImage.getCDim();

    V3DLONG VolumeXdim, VolumeYdim, VolumeZdim, VolumeCdim,Volumedatetype;
    VolumeXdim = (xe < originSize[0]) ? xe - xb : originSize[0] - xb;
    VolumeYdim = (ye < originSize[1]) ? ye - yb : originSize[1] - yb;
    VolumeZdim = (ze < originSize[2]) ? ze - zb : originSize[2] - zb;
    VolumeCdim = originSize[3];
    Volumedatetype = originImage.getDatatype();

    unsigned char *originImageData = originImage.getRawData();
    VolumeImage.setRawDataPointer(new unsigned char[Volumedatetype * VolumeXdim * VolumeYdim * VolumeZdim * VolumeCdim]);

    V3DLONG i = 0;
    for (V3DLONG j = 0; j<totalBytes;j++){
        if(Volumedatetype == 1){
            VolumeImage.getRawData()[i] = originImageData[j];
            i++;
        }
        else if (Volumedatetype == 2){
            unsigned char value8;
            int  value16;
            value8 = originImageData[j];
            j++;
            value16 =  value8 + originImageData[j]*256;
            VolumeImage.getRawData()[i] =value16*255/4096;  //12bit
            i++;
            //break;
        } 
    }
    /*
    for (V3DLONG c = 0; c < VolumeCdim; c++) {
        for (V3DLONG z = zb; z < zb + VolumeZdim; z++) {
            for (V3DLONG y = yb; y < yb + VolumeYdim; y++) {
                for (V3DLONG x = xb; x < xb + VolumeXdim; x++) {
                    unsigned char value8;
                    if(Volumedatetype == 1){
                        value8 = originImageData[x+y*originSize[0]+z*originSize[0]*originSize[1]+c*originSize[0]*originSize[1]*originSize[2]];
                        VolumeImage.getRawData()[i] = value8;
                        i++;
                    }
                    else if (Volumedatetype == 2){
                        int value16 = 0;
                        value8 = originImageData[x+y*originSize[0]+z*originSize[0]*originSize[1]+c*originSize[0]*originSize[1]*originSize[2]];
                        value16 += value8;
                        value8 = originImageData[x+y*originSize[0]+z*originSize[0]*originSize[1]+c*originSize[0]*originSize[1]*originSize[2]+1];
                        value16 += value8*256;
                        cout<<value16<<endl;
                        VolumeImage.getRawData()[i] = value16*255/4090; //12 bit
                        i++;
                    }
                }
            }
        }
    }
    */
    VolumeImage.setXDim(VolumeXdim);
    VolumeImage.setYDim(VolumeYdim);
    VolumeImage.setZDim(VolumeZdim);
    VolumeImage.setCDim(VolumeCdim);
    VolumeImage.setDatatype(V3D_UINT16);
    VolumeImage.setOriginX((double) xb);
    VolumeImage.setOriginY((double) yb);
    VolumeImage.setOriginZ((double) zb);
    //cout<<"VolumeImage.getTotalBytes() : "<<VolumeImage.getTotalBytes()<<endl;
    originImage.cleanExistData();
    return true;
}

bool normalization(Image4DSimple *image) {
    V3DLONG imageSize = image->getTotalUnitNumber();
    auto pRawData = image->getRawData();
    unsigned char maxValue = 0, minValue = 255;
    for (V3DLONG i = 0; i < imageSize; ++i) {
        maxValue = std::max(pRawData[i], maxValue);
        minValue = std::min(pRawData[i], minValue);
    }
    if (maxValue == minValue) {
        v3d_msg("max value = min value, unable to normalize", false);
        return false;
    }
	//cout<<"max: "<<int(maxValue)<<" min: "<<int(minValue);
    for (V3DLONG i = 0; i < imageSize; ++i) {
        pRawData[i] = int(pRawData[i] - minValue) * 255 / (maxValue - minValue);
    }
    return true;
}

bool MPL(Image4DSimple *image) {
    V3DLONG imageSizeX = image->getXDim();
    V3DLONG imageSizeY = image->getYDim();
    V3DLONG imageSizeZ = image->getZDim();
    V3DLONG imageSizeC = image->getCDim();
    //cout<<"imageSize: "<<imageSizeX<<" "<<imageSizeY<<" "<<imageSizeZ<<" "<<imageSizeC<<endl;
    auto pRawData = image->getRawData();
    unsigned char *MLPData = new unsigned char[imageSizeX*imageSizeY];
    unsigned char zmax = 0;
    unsigned char maxValue = 0;
    V3DLONG pannel_num = imageSizeX*imageSizeY;
	if(imageSizeX>256 || imageSizeX>256)
	{
		cout<<"ERROW: image size >256 "<<endl;
		cout<<imageSizeX<<" "<<imageSizeY<<endl;
		return FALSE;
	}
    for (V3DLONG x = 0; x < imageSizeX; x++) 
    {  
        for(V3DLONG y = 0; y < imageSizeY; y++)
        {   zmax = 0;
            for(V3DLONG z = 0; z < imageSizeZ; z++)
			{   
				zmax = max(pRawData[x*imageSizeX + y + z*pannel_num], zmax);
			}
            MLPData[x*imageSizeX+y] = zmax;
            maxValue = max(zmax,maxValue);
        }
    }
    if (maxValue == 0) return FALSE;
    image->deleteRawDataAndSetPointerToNull();
    //pRawData = image->getRawData();
    image->setNewRawDataPointer(MLPData);
    /*
    for (V3DLONG i = 0; i<pannel_num;i++)
    {
        cout<<" MLPdata: "<<MLPData[i]<<endl;
        cout<<"pRAWDATA: "<<pRawData[i]<<endl;
        pRawData[i] = MLPData[i]; 
    }
    */
    image->setZDim(1);     
    image->setXDim(imageSizeX);
    image->setYDim(imageSizeY);   
    
    //delete[] MLPData;
    
    return true;
}

