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


float branch_sig_test, branch_sig_train, branch_bkg_test, branch_bkg_train;
float branch_sig_test_weight, branch_sig_train_weight, branch_bkg_test_weight, branch_bkg_train_weight;
float evt_wgt;


void tree_2_hist_1()
{

	TFile *f1 = new TFile("compressed/compressed_BDT_all_mA_out_2018.root", "read");
	TFile *f2 = new TFile("training_root_files/WH_all_mA_2018_bdt_v1.root", "read");
	TFile *f3 = new TFile("training_root_files/TTbar_2018_bdt_v1.root", "read");

        TTree *Tout1 = (TTree*)f1->Get("Tree_sig_test");
	TTree *Tout2 = (TTree*)f1->Get("Tree_sig_train");
	TTree *Tout3 = (TTree*)f1->Get("Tree_bkg_test");
	TTree *Tout4 = (TTree*)f1->Get("Tree_bkg_train");

	TTree *Tout5 = (TTree*)f2->Get("tree");
	TTree *Tout6 = (TTree*)f3->Get("tree");

	Tout1->SetBranchAddress("branch_sig_test",&branch_sig_test);
        Tout2->SetBranchAddress("branch_sig_train",&branch_sig_train);
	Tout3->SetBranchAddress("branch_bkg_test",&branch_bkg_test);
        Tout4->SetBranchAddress("branch_bkg_train",&branch_bkg_train);
	Tout1->SetBranchAddress("branch_sig_test_weight",&branch_sig_test_weight);
        Tout2->SetBranchAddress("branch_sig_train_weight",&branch_sig_train_weight);
        Tout3->SetBranchAddress("branch_bkg_test_weight",&branch_bkg_test_weight);
        Tout4->SetBranchAddress("branch_bkg_train_weight",&branch_bkg_train_weight);

	Tout5->SetBranchAddress("evt_wgt",&evt_wgt);
	Tout6->SetBranchAddress("evt_wgt",&evt_wgt);

	int n;
        cout << "Enter bin no: " << endl;
        cin >> n;

	char name[1000];
        TFile *fout;
        sprintf(name,"compressed/compressed_BDT_all_mA_out_2018_bin%i.root",n);
        fout = new TFile(name,"RECREATE");

	TH1F* h_sig_test = new TH1F("h_sig_test", "h_sig_test", n, 0.0, 1.0);
        h_sig_test->Sumw2();
	TH1F* h_sig_train = new TH1F("h_sig_train", "h_sig_train", n, 0.0, 1.0);
        h_sig_train->Sumw2();
	TH1F* h_bkg_test = new TH1F("h_bkg_test", "h_bkg_test", n, 0.0, 1.0);
        h_bkg_test->Sumw2();
	TH1F* h_bkg_train = new TH1F("h_bkg_train", "h_bkg_train", n, 0.0, 1.0);
        h_bkg_train->Sumw2();


	int nevt1;
        nevt1=Tout1->GetEntries();

        for (int i = 0; i < nevt1; i++)
        {
                Tout1->GetEntry(i);
		Tout5->GetEntry(i);

		h_sig_test->Fill(branch_sig_test,evt_wgt/0.33);
        }

	int nevt2;
        nevt2=Tout2->GetEntries();

        for (int i = 0; i < nevt2; i++)
        {
                Tout2->GetEntry(i);
		Tout5->GetEntry(i);

		h_sig_train->Fill(branch_sig_train,evt_wgt/0.66);
        }

	int nevt3;
        nevt3=Tout3->GetEntries();

        for (int i = 0; i < nevt3; i++)
        {
                Tout3->GetEntry(i);
		Tout6->GetEntry(i);

		h_bkg_test->Fill(branch_bkg_test,evt_wgt/0.33);
        }

	int nevt4;
        nevt4=Tout4->GetEntries();

        for (int i = 0; i < nevt4; i++)
        {
                Tout4->GetEntry(i);
		Tout6->GetEntry(i);

		h_bkg_train->Fill(branch_bkg_train,evt_wgt/0.66);
        }


        f1->cd();
	f2->cd();
	f3->cd();
        delete Tout1;
	delete Tout2;
	delete Tout3;
	delete Tout4;
	delete Tout5;
	delete Tout6;
        delete f1;
	delete f2;
	delete f3;

        fout->cd();
        fout->Write();
        fout->Close();

}
