#include <iostream>
#include<fstream>
#include<string>
#include "TObject.h"
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TLorentzVector.h"
#include <ctime>


void sig_find_BDT_1D()
{

	TFile* f = TFile::Open("compressed/compressed_BDT_all_mA_out_2018_bin200.root");

	TH1F *h1 = (TH1F*) f->Get("h_sig_train");
        TH1F *h2 = (TH1F*) f->Get("h_bkg_train");

	int dim = h1->GetNbinsX();
	float significance[dim];

	for(int i = 1; i < dim; i++)
	{

		float sum = 0.0;
		float bkg = 0.0;

		for(int j = i+1; j <= dim; j++)
		{
			float s = h1->GetBinContent(j);
                	float b = h2->GetBinContent(j);

			sum = sum + s;
			bkg = bkg + b;
		}

		if ((sum+bkg) <= 0) continue;

		significance[i] = sum/sqrt(sum+bkg);

		cout << (1.0/dim)*i << "		" << significance[i] << endl;

	}

	float max = -999.0;
        int index = -1;

        for (int i = 1; i < dim; i++)
        {
                if(significance[i]>max)
                {
                        max = significance[i];
                        index = i;
                }
        }


        cout << "max sensitivity : " << max << endl;
	cout << "corresponding BDT score : " << (1.0/dim)*index << endl;
        cout << "corresponding bin no : " << index << endl;

}
