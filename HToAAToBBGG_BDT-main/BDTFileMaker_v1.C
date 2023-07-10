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


using namespace std;

double PhiInRange(const double& phi) {
  double phiout = phi;

  if( phiout > 2*M_PI || phiout < -2*M_PI) {
    phiout = fmod( phiout, 2*M_PI);
  }
  if (phiout <= -M_PI) phiout += 2*M_PI;
  else if (phiout >  M_PI) phiout -= 2*M_PI;

  return phiout;
}

double delta2R(double eta1, double phi1, double eta2, double phi2) {
  return sqrt(pow(eta1 - eta2,2) +pow(PhiInRange(phi1 - phi2),2));
}


void BDTFileMaker_v1()
{

	float Leppt, Lepeta, Lepphi, Lepe, MetPhi;
	float B1pt, B1eta, B1phi, B1e;
	float B2pt, B2eta, B2phi, B2e;
	float bb_inv_mass, b1_DeepFlv, b2_DeepFlv;
	float Pho1pt, Pho1eta, Pho1phi, Pho1e, pho1MVA;
	float Pho2pt, Pho2eta, Pho2phi, Pho2e, pho2MVA;
	float dipho_invmass, invmassbbgg, delphi_ggMET, delphi_bb, delphi_gg, delphi_bbgg;
	double weight_nom;
	bool isEle, isMu;
	int ab, bb;


	float leppt, lepeta, lepphi, lepe;
        float b1pt, b1eta, b1phi, b1e;
        float b2pt, b2eta, b2phi, b2e;
        float leadbdeepjet, subleadbdeepjet;
        float pho1pt, pho1eta, pho1phi, pho1e, leadgMVA, pT1_by_mgg, pT1_by_mbb;
        float pho2pt, pho2eta, pho2phi, pho2e, subleadgMVA, pT2_by_mgg, pT2_by_mbb;
        float m_bb, m_gg, m_bbgg;
        float evt_wgt;
	int yout;
	int jet_no, Njets;
	

	TFile *fout = new TFile("training_root_files/DYJetsToLLM50V2_2018_bdt_v1.root","RECREATE");
        TTree *tree = new TTree("tree","tree");

   	tree->Branch("leppt",&leppt);
	tree->Branch("lepeta",&lepeta);
	
	tree->Branch("b1pt",&b1pt);
        tree->Branch("b1eta",&b1eta);
	tree->Branch("b2pt",&b2pt);
        tree->Branch("b2eta",&b2eta);

	tree->Branch("pho1pt",&pho1pt);
        tree->Branch("pho1eta",&pho1eta);
        tree->Branch("pho2pt",&pho2pt);
        tree->Branch("pho2eta",&pho2eta);

	tree->Branch("m_bb",&m_bb);
	tree->Branch("m_gg",&m_gg);
	tree->Branch("m_bbgg",&m_bbgg);
	tree->Branch("pT1_by_mgg",&pT1_by_mgg);
	tree->Branch("pT2_by_mgg",&pT2_by_mgg);
	tree->Branch("pT1_by_mbb",&pT1_by_mbb);
        tree->Branch("pT2_by_mbb",&pT2_by_mbb);
	tree->Branch("Njets",&Njets);

	tree->Branch("leadbdeepjet",&leadbdeepjet);
	tree->Branch("subleadbdeepjet",&subleadbdeepjet);
	tree->Branch("leadgMVA",&leadgMVA);
	tree->Branch("subleadgMVA",&subleadgMVA);

	tree->Branch("delphi_ggMET",&delphi_ggMET);
	tree->Branch("delphi_gg",&delphi_gg);
	tree->Branch("delphi_bb",&delphi_bb);
	tree->Branch("delphi_bbgg",&delphi_bbgg);

	tree->Branch("evt_wgt",&evt_wgt);
	tree->Branch("yout",&yout);


	TFile *f = new TFile("/home/anirban/root_files/CR_studies/new_v1/MVA_modified/WH_2018/full_reg/full_ver_1/DYJetsToLLM50V2.root");
        TTree *tr = (TTree*)f->Get("Tout");

        tr->SetBranchAddress("leppt",&Leppt);
        tr->SetBranchAddress("lepeta",&Lepeta);

	tr->SetBranchAddress("MetPhi",&MetPhi);

        tr->SetBranchAddress("b1pt",&B1pt);
        tr->SetBranchAddress("b1eta",&B1eta);
	tr->SetBranchAddress("b1phi",&B1phi);
	tr->SetBranchAddress("b1e",&B1e);
        tr->SetBranchAddress("b2pt",&B2pt);
        tr->SetBranchAddress("b2eta",&B2eta);
	tr->SetBranchAddress("b2phi",&B2phi);
        tr->SetBranchAddress("b2e",&B2e);

        tr->SetBranchAddress("pho1pt",&Pho1pt);
        tr->SetBranchAddress("pho1eta",&Pho1eta);
        tr->SetBranchAddress("pho1phi",&Pho1phi);
	tr->SetBranchAddress("pho1e",&Pho1e);
        tr->SetBranchAddress("pho2pt",&Pho2pt);
        tr->SetBranchAddress("pho2eta",&Pho2eta);
        tr->SetBranchAddress("pho2phi",&Pho2phi);
	tr->SetBranchAddress("pho2e",&Pho2e);

        tr->SetBranchAddress("dipho_invmass",&dipho_invmass);
        tr->SetBranchAddress("invmassbbgg",&invmassbbgg);
	tr->SetBranchAddress("bb_inv_mass",&bb_inv_mass);

	tr->SetBranchAddress("b1_DeepFlv",&b1_DeepFlv);
	tr->SetBranchAddress("b2_DeepFlv",&b2_DeepFlv);
	tr->SetBranchAddress("pho1MVA",&pho1MVA);
	tr->SetBranchAddress("pho2MVA",&pho2MVA);
	tr->SetBranchAddress("jet_no",&jet_no);

	tr->SetBranchAddress("isMu",&isMu);
	tr->SetBranchAddress("isEle",&isEle);
	tr->SetBranchAddress("ab",&ab);
	tr->SetBranchAddress("bb",&bb);
	tr->SetBranchAddress("weight_nom",&weight_nom);


	double TTGJets = (59730*4.078)/(2.66557e+07);
	double DY = (59730*5343)/(9.14154e+07);
	double TTSL = (59730*365.34)/(1.48113e+11);
	double TT2L2Nu = (59730*88.29)/(1.03533e+10);
	double WH_mA20 = (59730*0.01)/(993834);
	double WH_mA25 = (59730*0.01)/(993842);
	double WH_mA30 = (59730*0.01)/(927844);
	double WH_mA35 = (59730*0.01)/(993806);
	double WH_mA40 = (59730*0.01)/(969827);
	double WH_mA45 = (59730*0.01)/(948824);
	double WH_mA50 = (59730*0.01)/(720881);
	double WH_mA55 = (59730*0.01)/(903853);
	double WH_mA60 = (59730*0.01)/(807874);


	for(int ij=0; ij<tr->GetEntries(); ++ij){
		tr->GetEntry(ij);

		if((isEle || isMu) && (bb == 1) && (weight_nom > 0)) {
		leppt = Leppt;
		lepeta = Lepeta;
		b1pt = B1pt;
		b1eta = B1eta;
		b2pt = B2pt;
                b2eta = B2eta;
		pho1pt = Pho1pt;
		pho1eta = Pho1eta;
		pho2pt = Pho2pt;
                pho2eta = Pho2eta;

		TLorentzVector g1, g2;
                g1.SetPtEtaPhiE(Pho1pt, Pho1eta, Pho1phi, Pho1e);
                g2.SetPtEtaPhiE(Pho2pt, Pho2eta, Pho2phi, Pho2e);
		delphi_gg = fabs(g1.DeltaPhi(g2));
                delphi_ggMET = fabs(PhiInRange((g1+g2).Phi() - MetPhi));

		TLorentzVector b1, b2;
                b1.SetPtEtaPhiE(B1pt, B1eta, B1phi, B1e);
                b2.SetPtEtaPhiE(B2pt, B2eta, B2phi, B2e);
		delphi_bb = fabs(b1.DeltaPhi(b2));
		delphi_bbgg = fabs((b1+b2).DeltaPhi(g1+g2));

		m_bb = bb_inv_mass;
		m_gg = dipho_invmass;
		m_bbgg = invmassbbgg;
		pT1_by_mgg = Pho1pt/dipho_invmass;
		pT2_by_mgg = Pho2pt/dipho_invmass;
		pT1_by_mbb = B1pt/bb_inv_mass;
                pT2_by_mbb = B2pt/bb_inv_mass;
		Njets = jet_no;

		leadbdeepjet = b1_DeepFlv;
		subleadbdeepjet = b2_DeepFlv;
		leadgMVA = pho1MVA;
		subleadgMVA = pho2MVA;

		evt_wgt = weight_nom*DY;
		yout = 0;

		tree->Fill(); }

	}

	fout->cd();
	tree->Write();
	fout->Close();

}
