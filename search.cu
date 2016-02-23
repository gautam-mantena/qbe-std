#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <math.h>
#include <limits>
#include <time.h>
#include "cublas.h"


using namespace std;

class Path {
    /*
    This class is used to store the path information of the search hits
    */
    public:
    string file_id;
    int first_query,last_query;
    int first_ref,last_ref;
    double score;
    int count;

};

class Speech{
    public:
    double *speech;
    int speechSize;
};



__global__ void cudaLogDot(double *ar) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	ar[idx] = -log(ar[idx]);
}


float compute_overlap(int startx, int endx, int starty, int endy) {
    /*
    It gives the amount of overlap between the two detected search hits
    */
    int min_end,max_start,min_size;
    int xsize,ysize;

    if (endx < endy) {
        min_end = endx;
    }
    else {
        min_end = endy;
    }

    if (startx > starty) {
        max_start = startx;
    }
    else {
        max_start = starty;
    }

    xsize = endx - startx;
    ysize = endy - starty;

    if (xsize < ysize) {
        min_size = xsize;
    }
    else {
        min_size = ysize;
    }

    return (min_end - max_start)/float(min_size);
}

vector<Path> merge_paths(vector<Path> matching_paths, float minOverlap) {
    /*
    Merging all the paths from the detected search hits based on a minOverlap
    */
    vector<Path> merge;

    if (matching_paths.size() == 0) {
        return merge;
    }

    merge.push_back(matching_paths.at(0));
    for (int i = 1; i < matching_paths.size();i++) {
        int merge_flag = 0;
        for (int j = 0; j < merge.size();j++) {
            if (merge.at(j).file_id.compare(matching_paths.at(i).file_id) == 0) {
                float overlap1 = compute_overlap(matching_paths.at(i).first_query,matching_paths.at(i).last_query,merge.at(j).first_query,merge.at(j).last_query);

                float overlap2 = compute_overlap(matching_paths.at(i).first_ref,matching_paths.at(i).last_ref,merge.at(j).first_ref,merge.at(j).last_ref);

                if (overlap1 > minOverlap && overlap2 > minOverlap) {
                    merge_flag = 1;
                    if (merge.at(j).score > matching_paths.at(i).score) {
                        merge.at(j) = matching_paths.at(i);
                    }
                    break;
                }

            }
        }
        if (!merge_flag) {
            merge.push_back(matching_paths.at(i));
        }
    }
    return merge;
}

class Feats {
    public:
    void get_files(char *fname, vector<string> &featFiles, vector<string> &sadFiles) {
        ifstream inFile(fname,ios::in);

        if (! inFile) {
            cout << "No Such File: " << fname << endl;
            exit(1);
        }

        string featF;
        string sadF;

        inFile >> featF >> sadF;

        while (! inFile.eof()) {
            featFiles.push_back(featF);
            sadFiles.push_back(sadF);
            inFile >> featF >> sadF;
        }
    }

    void load_sad(vector<string> sadFiles, vector<vector <int> > &sadInfo) {
        cout << "Loading SAD files" << endl;

       for (int i = 0; i < sadFiles.size(); i++ ) {

            cout << "Load: " << sadFiles.at(i) << endl;
            vector<int> sad;
            ifstream inFile((char *)sadFiles.at(i).c_str(),ios::in);

            if (!inFile) {
                cout << "No such file: "<<sadFiles.at(i) << endl;
                exit(1);
            }

            char sad_value;
            inFile >> sad_value;
            while (!inFile.eof()) {
                sad.push_back(atoi(&sad_value));
                inFile >> sad_value;
            }
            sadInfo.push_back(sad);
            inFile.close();
        }

    }

    void load_ref(vector<string> refFiles, vector<vector<int> > ref_sadInfo, vector<vector<double * > > &refFeats, int SAD, int dim) {
        cout << "Loading feats" << endl;
        for (int i=0;i < refFiles.size();i++) {
            cout << "Load: " << refFiles.at(i) << endl;
            ifstream inFile((char *)refFiles.at(i).c_str(),ios::in);

            if (! inFile) {
                cout << "No such file: " << refFiles.at(i) << endl;
                exit(1);
            }

            //cannot count on the sad file. Only consider the mfcc feats.
            int row=0;
            double data;
            inFile >> data;
            vector<double *> feat_array;
            while (! inFile.eof()) {
                if (ref_sadInfo.at(i).at(row) > SAD) {
                    double *feats = (double *) malloc(dim * sizeof(double));
                    for (int j = 0; j < dim; j++) {
                        feats[j] = data;
                        inFile >> data;
                    }
                    feat_array.push_back(feats);
                }
                else {
                    for (int j = 0; j < dim; j++) {
                        inFile >> data;
                    }
                }
                row += 1;
            }
            refFeats.push_back(feat_array);
            inFile.close();
        }
    }

    void get_feats(string feat_fname, string sad_fname, vector<double * > &feat_array, int SAD, int dim) {
        //reading sad_name
        ifstream inFile((char *)sad_fname.c_str(),ios::in);
        vector<double> sadInfo;

        if (!inFile) {
            cout << "No such file: " << sad_fname << endl;
            exit(1);
        }

        char data;
        inFile >> data;
        while (!inFile.eof()) {
            sadInfo.push_back(atoi(&data));
            inFile >> data;
        }
        inFile.close();

        //cannot count on the sad file. Only consider the mfcc feats.
        int row=0;
        double db_data;
        inFile.open((char *)feat_fname.c_str(),ios::in);
        inFile >> db_data;
        while (! inFile.eof()) {
            if (sadInfo.at(row) > SAD) {
                double *feats = (double *) malloc(dim * sizeof(double));
                for (int j = 0; j < dim; j++) {
                    feats[j] = db_data;
                    inFile >> db_data;
                }
                feat_array.push_back(feats);
            }
            else {
                for (int j = 0; j < dim; j++) {
                        inFile >> db_data;
                }
            }
            row += 1;
        }
        inFile.close();
    }

    void undo_sad(vector<int> ref_sad, Path &path, int SAD) {
        int buf = 0;
        int pos = 0;
        for (int i = 0; i < ref_sad.size();i++) {
            if (pos >= path.first_ref) {
                break;
            }

            if (ref_sad.at(i) <= SAD) {
                buf += 1;
            }
            else {
                pos += 1;
            }
        }

        path.first_ref += buf;
        buf = 0;
        pos = 0;
        for (int i = 0; i < ref_sad.size();i++) {
            if (pos >= path.last_ref) {
                break;
            }

            if (ref_sad.at(i) <= SAD) {
                buf += 1;
            }
            else {
                pos += 1;
            }
        }
        path.last_ref += buf;
    }

    void getRefFlats(vector<Speech> &refFlat, vector<vector<double *> > refFeats, int dim) {
        for (int i = 0; i < refFeats.size(); i++) {
            vector<double *> ref = refFeats.at(i);
            Speech refInfo;
            refInfo.speechSize = ref.size();
            refInfo.speech = (double *) malloc(sizeof(double) * dim * refInfo.speechSize);
            for (int j = 0; j < refInfo.speechSize; j++) {
                for (int k = 0; k < dim; k++) {
                    int index = j + (k * refInfo.speechSize);
                    refInfo.speech[index] = ref.at(j)[k];
                }
            }
            refFlat.push_back(refInfo);
        }

    }

    void getQueryFlat(Speech &queryInfo, vector<double *> query, int dim) {
        int querySize = query.size();
        queryInfo.speechSize = querySize;
        queryInfo.speech = (double *) malloc(sizeof(double) * dim * queryInfo.speechSize);

        for (int i = 0; i < queryInfo.speechSize; i++) {
            for (int j = 0; j < dim; j++) {
                int index = i + (j * queryInfo.speechSize);
                queryInfo.speech[index] = query.at(i)[j];
            }
        }
    }
};


class DTW {
    public:

    double euc(double *v1, double *v2, int dim) {
        double distance = 0.0;
        for (int i = 0; i < dim; i++) {
            distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        }
        return sqrt(distance);
    }

    double logdot(double *v1,double *v2,int dim) {
        double distance = 0.0;
        for (int i = 0;i < dim; i++) {
            distance += v1[i] * v2[i];
        }
        return -1.0 * log(distance);
    }

    vector<Path> CUDA_NS_DTW(double *query, int querySize, double *ref, int refSize, int dim, int nbest) {
        vector<Path> matching_paths;
        double *flatMatrix, *flatJumps, *flatOrigin;
        flatMatrix = (double *) malloc(sizeof(double) * refSize * querySize);
        flatJumps = (double *) malloc(sizeof(double) * refSize * querySize);
        flatOrigin = (double *) malloc(sizeof(double) * refSize * querySize);


        //INIT CUDA RELATED CODE


        cublasStatus_t status;

        //***********************START ALLOCATE ARRAYS********************************************
        double *cudaQuery, *cudaRef, *cudaMatrix;

        status = cublasAlloc(querySize * dim, sizeof(double), (void **)&cudaQuery);
        if (status != CUBLAS_STATUS_SUCCESS) {
                cout << "CANNOT INIT cudaQuery"  << endl;
                exit(1);
        }

        status = cublasAlloc(refSize * dim, sizeof(double), (void **) &cudaRef);
        if (status != CUBLAS_STATUS_SUCCESS) {
                cout << "CANNOT INIT cudaRef" << endl;
                exit(1);
        }

        status = cublasAlloc(refSize * querySize, sizeof(double), (void **) &cudaMatrix);
        if (status != CUBLAS_STATUS_SUCCESS) {
                cout << "CANNOT INIT cudaMatrix" << endl;
                exit(1);
        }
        //***********************END ALLOCATE ARRAYS**********************************************

        //***********************START COPY TO DEVICE*********************************************
        status = cublasSetMatrix(querySize, dim, sizeof(double), query,  querySize, cudaQuery, querySize);
        if (status != CUBLAS_STATUS_SUCCESS) {
                cout << "CANNOT COPY flatQuery" << endl;
                exit(1);
        }

        status = cublasSetMatrix(refSize, dim, sizeof(double), ref, refSize, cudaRef, refSize);
        if (status != CUBLAS_STATUS_SUCCESS) {
                cout << "CANNOT COPY FLATREF" << endl;
                exit(1);
        }
        //**************i*********END COPY TO DEVICE***********************************************

         //***********************START MATRIX MULTIPLICATION**************************************
        cublasDgemm('n', 't', refSize, querySize, dim, 1.0, cudaRef, refSize, cudaQuery, querySize, 0.0 , cudaMatrix, refSize );
        status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS) {
                cout << "ERROR MATRIX MULTIPLICATION" << endl;
                exit(1);
        }
	
	//performing log operation
	cudaLogDot<<<refSize,querySize>>>(cudaMatrix);

        status = cublasGetMatrix(refSize, querySize, sizeof(double), cudaMatrix, refSize, flatMatrix, refSize);
        if (status != CUBLAS_STATUS_SUCCESS) {
                cout << "ERROR COPY C" << endl;
                exit(1);
        }

	

        for(int i = 0; i < refSize; i++) {
            for (int j = 0; j < querySize; j++) {
                int index = i + (j * refSize);
		//flatMatrix[index] = -log(flatMatrix[index]);
		flatMatrix[index] = flatMatrix[index];
                double distance = flatMatrix[index];
                if (j == 0) {
                    flatJumps[index] = 2.0;
                    flatMatrix[index] = flatMatrix[index]/flatJumps[index];
                    flatOrigin[index] = i;
                }
                else if (i == 0) {
                    int bot_index = index - refSize;
                    flatMatrix[index] = (flatMatrix[bot_index] * flatJumps[bot_index]) + distance;
                    flatJumps[index] = flatJumps[bot_index] + 1;
                    flatMatrix[index] = flatMatrix[index]/flatJumps[index];
                    flatOrigin[index] = flatOrigin[bot_index];
                }
                else if (i == 1) {
                    int bot_index = index - refSize;
                    int mid_index = bot_index - 1;

                    double mid, bot;
                    mid = ((flatMatrix[mid_index] * flatJumps[mid_index]) + distance)/(flatJumps[mid_index] + 2);
                    bot = ((flatMatrix[bot_index] * flatJumps[bot_index]) + distance)/(flatJumps[bot_index] + 1);

                    if (mid <= bot) {
                        flatMatrix[index] = mid;
                        flatJumps[index] = flatJumps[mid_index] + 2;
                        flatOrigin[index] = flatOrigin[mid_index];
                    }
                    else {
                        flatMatrix[index] = bot;
                        flatJumps[index] = flatJumps[bot_index] + 1;
                        flatOrigin[index] = flatOrigin[bot_index];
                    }
                }
                else {
                    int bot_index = index - refSize;
                    int mid_index = bot_index - 1;
                    int top_index = bot_index - 2;

                    double mid, bot, top;
                    top = ((flatMatrix[top_index] * flatJumps[top_index]) + distance)/(flatJumps[top_index] + 1);
                    mid = ((flatMatrix[mid_index] * flatJumps[mid_index]) + distance)/(flatJumps[mid_index] + 2);
                    bot = ((flatMatrix[bot_index] * flatJumps[bot_index]) + distance)/(flatJumps[bot_index] + 1);

                    if (mid <= top && mid <= bot) {
                        flatMatrix[index] = mid;
                        flatJumps[index] = flatJumps[mid_index] + 2;
                        flatOrigin[index] = flatOrigin[mid_index];
                    }
                    else if (top <= mid && top <= bot) {
                        flatMatrix[index] = top;
                        flatJumps[index] = flatJumps[top_index] + 1;
                        flatOrigin[index] = flatOrigin[top_index];
                    }
                    else {
                        flatMatrix[index] = bot;
                        flatJumps[index] = flatJumps[bot_index] + 1;
                        flatOrigin[index] = flatOrigin[bot_index];
                    }



                }
            }
        }
	
	//printing the values delete it later
	/*for (int i = 0; i < refSize; i++) {
		for (int j = 0; j < querySize; j++) {
			int index = i + (j * refSize);
			cout << flatMatrix[index] << " ";
		}
		cout << endl;
	}
	exit(1);*/



        double MAX_VALUE = numeric_limits<double>::max( );

        //retrieving the paths
        for (int n = 0; n < nbest; n++) {

            double bestScore = MAX_VALUE;
            int start_index,end_index = -1;
            int start_query=0,end_query=querySize - 1;

            int buffer_index = (querySize - 1) * refSize;

            for (int i = 0; i < refSize; i++) {
                if (bestScore > flatMatrix[buffer_index + i]) {
                    bestScore = flatMatrix[buffer_index + i];
                    end_index = buffer_index + i;
                }
            }

            if (end_index == -1) {
                break;
            }

            start_index = flatOrigin[end_index];

            //removing paths around the best path
            int sur_start = 0,sur_end = 0;

            if (end_index % refSize == 0) {
                sur_start = end_index;
            }
            else {
                sur_start = end_index - 1;
            }

            if (end_index % refSize == querySize) {
                sur_end = end_index;
            }
            else {
                sur_end = end_index + 1;
            }

            while (sur_start % refSize !=0 && flatMatrix[sur_start] > flatMatrix[sur_start+1]) {
                sur_start = sur_start - 1;
            }

            while (sur_end % refSize != querySize && flatMatrix[sur_end] > flatMatrix[sur_end-1]) {
                sur_end = sur_end + 1;
            }


            for (int i=sur_start; i < sur_end; i++) {
                flatMatrix[i] = MAX_VALUE;
            }

            int start_ref = start_index;
            int end_ref = end_index - ((querySize-1) * refSize);


            //checking if its a valid path
            if (start_ref != end_ref) {
                Path path_obj;
                path_obj.file_id = "FILE_ID";
                path_obj.first_query = start_query;
                path_obj.last_query = end_query;
                path_obj.first_ref = start_ref;
                path_obj.last_ref = end_ref;
                path_obj.score = bestScore;
                path_obj.count = 1;
                matching_paths.push_back(path_obj);
            }
        }

        //array freeing
        free(flatMatrix);
        free(flatJumps);
        free(flatOrigin);

        cublasFree(cudaQuery);
        cublasFree(cudaRef);
        cublasFree(cudaMatrix);

         //cublasShutdown();

        return matching_paths;
    }


};




int main(int argc, char **argv) {

    //HARD CODED PARAMETERS FOR THE ALGORITHM
    int dim = 128, SAD = 1,nbest=1;
    float minOverlap = 0.5;
    float frameRate = 10; //10 ms
    int SEARCH_LIMIT = 1660;

    if (argc == 1) {
        cout << "./search <queryList> <refList>" << endl;
        exit(1);
    }
    
    //init CUDA
    cublasInit();

    char *queryFileList = argv[1];
    char *refFileList = argv[2];

    cout << "Query: " << queryFileList << endl;
    cout << "Reference: " << refFileList << endl;

    Feats feat_obj;

    //reading reference files
    vector<string> refFiles,ref_sadFiles;
    feat_obj.get_files(refFileList,refFiles,ref_sadFiles);

    //loading reference sad files
    vector<vector<int> > ref_sadInfo;
    feat_obj.load_sad(ref_sadFiles,ref_sadInfo);

    //load ref files
    vector<vector<double *> > refFeats;
    vector<int> refSizes;
    feat_obj.load_ref(refFiles, ref_sadInfo, refFeats, SAD, dim);
    cout << "Flat Ref Arrays" << endl;
    vector<Speech> refFlats;
    feat_obj.getRefFlats(refFlats, refFeats, dim);

    //read query files
    vector<string> queryFiles,query_sadFiles;
    feat_obj.get_files(queryFileList,queryFiles,query_sadFiles);

    //calculating the time for the whole search process
    clock_t startTime = clock();

    //search starts from here
    DTW dtw_obj;
    for (int i = 0; i < queryFiles.size(); i++) {
        //get query feats.
        vector<double *> query;
        feat_obj.get_feats(queryFiles.at(i), query_sadFiles.at(i), query, SAD, dim);
        Speech queryInfo;
        feat_obj.getQueryFlat(queryInfo, query, dim);

        cout << "Query: " << queryFiles.at(i) << endl;
        int results = 1;

        vector<Path> best_paths;
        int max_index = 0;

        for (int j = 0; j < refFlats.size();j++) {
            Speech refInfo = refFlats.at(j);
            vector<Path> matching_path;
            matching_path = dtw_obj.CUDA_NS_DTW(queryInfo.speech, queryInfo.speechSize, refInfo.speech, refInfo.speechSize, dim, nbest);

            matching_path = merge_paths(matching_path, minOverlap);
            for (int n = 0; n < matching_path.size();n++) {
                feat_obj.undo_sad(ref_sadInfo.at(j), matching_path.at(n),SAD);
                matching_path.at(n).file_id = refFiles.at(j);
            }

            //-------------------------------------------------------------------------
            for (int k = 0; k < matching_path.size(); k++) {
                if (best_paths.size() < SEARCH_LIMIT - 1) {
                    best_paths.push_back(matching_path.at(k));
                }
                else if (best_paths.size() == SEARCH_LIMIT - 1) {
                    best_paths.push_back(matching_path.at(k));

                    max_index = 0;

                    for (int t=1;t < SEARCH_LIMIT; t++) {
                        if (best_paths.at(t).score > best_paths.at(max_index).score) {
                            max_index = t;
                        }
                    }
                }
                else {

                    if (matching_path.at(k).score < best_paths.at(max_index).score) {
                        best_paths.at(max_index) = matching_path.at(k);
                        for (int t = 0; t < SEARCH_LIMIT; t++) {
                            if (best_paths.at(t).score > best_paths.at(max_index).score) {
                                max_index = t;
                            }
                        }
                    }
                }
            }
            //-------------------------------------------------------------------------

        }

        if (best_paths.size() != 0) {
            for (int n = 0; n < best_paths.size(); n++) {
                cout << "RESULT: Query" << " " << queryFiles.at(i) << " " << "ref" << " " << best_paths.at(n).file_id << " " << "MATCH_FOUND score" << " " << best_paths.at(n).score << " " << "query" << " " << best_paths.at(n).first_query * (frameRate/1000.0) << " " << best_paths.at(n).last_query * (frameRate/1000.0) << " " << "ref" << " " << best_paths.at(n).first_ref * (frameRate/1000.0) << " " << best_paths.at(n).last_ref * (frameRate/1000.0) << endl;

            }
        }
        else {
            cout << "RESULT: Query" << " " << queryFiles.at(i) << " " << "ref ALL_FILES NO_MATCH score 0";
        }
    }

    clock_t endTime = clock();
    double searchTime = double(endTime - startTime);
    searchTime = searchTime/(double)CLOCKS_PER_SEC;

    cout << "Total Search Time (in sec): " << searchTime << endl;


    return 0;
}




