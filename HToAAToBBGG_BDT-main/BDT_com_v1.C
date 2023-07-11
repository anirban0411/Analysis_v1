#include <iostream>
#include <fstream>
#include <set>
#include<TFile.h>
#include<TTree.h>
#include<TH1.h>
#include<TH2.h>
#include<THStack.h>
#include<TGraphErrors.h>
#include<TGraphAsymmErrors.h>
#include<TCanvas.h>
#include<TFrame.h>
#include<TLegend.h>
#include<vector>
#include<iostream>
#include<TMath.h>
#include<TROOT.h>
#include<TInterpreter.h>
#include<TStyle.h>
#include<TChain.h>
#include<TString.h>
#include<TPaveStats.h>
#include<TPad.h>
#include<TLatex.h>
#include "TEfficiency.h"
#include "TAxis.h"
//#include "TOTTOTerror.h"
//#include "JERerror.h"
//#include "JESerror.h"
#include "staterror.h"


void BDT_com_v1(){
  Bool_t syst_on = false;
  gStyle->SetLineWidth(2);
  TCanvas *c1 =new TCanvas("c1", " ", 0, 0,700,800);

  c1->Range(0,0,1,1);
  c1->SetFillColor(0);
  c1->SetBorderMode(0);
  c1->SetBorderSize(2);
  c1->SetFrameBorderMode(0);
  c1->SetGridx();
  c1->SetGridy();
  c1->SetTickx(1);
  c1->SetTicky(1);
  c1->Draw();

//  c1->SetLogy();

  TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1);
  pad1->SetBottomMargin(0.02);
  pad1->Draw();
  pad1->cd();
  pad1->SetLogy();
  pad1->SetGridx();
  pad1->SetGridy();
  pad1->SetTickx(1);
  pad1->SetTicky(1);


  double lumi_18 = 59730;

  THStack *hs = new THStack("hs","");
  THStack *hs_mod = new THStack("hs_mod","");

  Double_t error;
  ////////////////////Data rootfiles/////////////////////

  TFile *fdata1 = TFile::Open("compressed/compressed_BDT_SingleMuon_out_2018_bin50.root");
  TH1F  *h1 = (TH1F*)fdata1->Get("h_bkg_test");
  TFile *fdata2 = TFile::Open("compressed/compressed_BDT_Egamma_out_2018_bin50.root");
  TH1F  *h2 = (TH1F*)fdata2->Get("h_bkg_test");
  h1->Add(h2);
  h1->SetBinContent(h1->GetNbinsX(), h1->GetBinContent(h1->GetNbinsX())+h1->GetBinContent(h1->GetNbinsX()+1));
  h1->SetBinContent(1, h1->GetBinContent(1)+h1->GetBinContent(0));


  int n = h1->GetNbinsX();
  float q = (1.-0.6827)/2.;
  float N, dN, yscale;
  Double_t x[n], y[n], EYlow[n], EYhigh[n], EXlow[n], EXhigh[n];
  for(int i = 0; i < n; i++){
    N = h1->GetBinContent(i+1);
    dN = h1->GetBinError(i+1);
    if(N > 0. && dN > 0. && abs((dN*dN)/N-1) > 0.0001){
      yscale = ((dN*dN)/N);
      N = (N/dN)*(N/dN);
    }
    else yscale = 1;
    x[i] = h1->GetXaxis()->GetBinCenter(i+1);
    y[i] = yscale*N;
    if(N > 0) EYlow[i] = yscale*(N - ROOT::Math::chisquared_quantile_c(1-q,2*N)/2.);
    else EYlow[i] = 0.;
    EYhigh[i] = yscale*(ROOT::Math::chisquared_quantile_c(q,2*(N+1))/2.- N);
    EXlow[i] = h1->GetXaxis()->GetBinUpEdge(i+1) - x[i];
    EXhigh[i] = x[i] - h1->GetXaxis()->GetBinLowEdge(i+1);
  }
  TGraphAsymmErrors *hdata1 = new TGraphAsymmErrors(n,x,y,EXlow,EXhigh,EYlow,EYhigh);

  double xmax = h1->GetXaxis()->GetXmax();
  double xmin = h1->GetXaxis()->GetXmin();
  double xmin1 = 0.0;
  double xmax1 = 1.0;

  //////////////////////MC files//////////////////////


  double w1 = (59730*4.078)/(1.40e+06);
  TFile *f1 = TFile::Open("compressed/compressed_BDT_TTGJets_out_2018_bin50.root");
  TH1F *hmc1 = (TH1F*)f1->Get("h_bkg_test");
  hmc1->SetBinContent(hmc1->GetNbinsX(), hmc1->GetBinContent(hmc1->GetNbinsX())+hmc1->GetBinContent(hmc1->GetNbinsX()+1));
  hmc1->SetBinContent(1, hmc1->GetBinContent(1)+hmc1->GetBinContent(0));
  hmc1->Scale(w1);

  double w2 = (59730*365.34)/(1.62e+08);
  TFile *f2 = TFile::Open("compressed/compressed_BDT_TTSL_out_2018_bin50.root");
  TH1F *hmc2 = (TH1F*)f2->Get("h_bkg_test");
  hmc2->SetBinContent(hmc2->GetNbinsX(), hmc2->GetBinContent(hmc2->GetNbinsX())+hmc2->GetBinContent(hmc2->GetNbinsX()+1));
  hmc2->SetBinContent(1, hmc2->GetBinContent(1)+hmc2->GetBinContent(0));
  hmc2->Scale(w2);

  double w3 = (59730*88.29)/(0.47e+08);
  TFile *f3 = TFile::Open("compressed/compressed_BDT_TTTo2L2Nu_out_2018_bin50.root");
  TH1F *hmc3 = (TH1F*)f3->Get("h_bkg_test");
  hmc3->SetBinContent(hmc3->GetNbinsX(), hmc3->GetBinContent(hmc3->GetNbinsX())+hmc3->GetBinContent(hmc3->GetNbinsX()+1));
  hmc3->SetBinContent(1, hmc3->GetBinContent(1)+hmc3->GetBinContent(0));
  hmc3->Scale(w3);


  TH1F *totalMC = (TH1F*)hmc1->Clone();
  totalMC->Add(hmc2);
  totalMC->Add(hmc3);

  double scaletodata = h1->Integral()/totalMC->Integral();

  totalMC->SetFillColor(14);
  gStyle->SetHatchesSpacing(0.7);
  gStyle->SetHatchesLineWidth(1.);
  totalMC->SetFillStyle(3354);

  hmc1->Scale(scaletodata);
  hmc1->Scale(0.9);
  hmc1->SetFillColor(kAzure+2);
  hmc1->SetFillStyle(1001);
  hmc1->SetLineColor(kBlack);

  hmc2->Scale(scaletodata);
  hmc2->Scale(0.9);
  hmc2->SetFillColor(kTeal-9);
  hmc2->SetFillStyle(1001);
  hmc2->SetLineColor(kBlack);

  hmc3->Scale(scaletodata);
  hmc3->Scale(0.9);
  hmc3->SetFillColor(kYellow-9);
  hmc3->SetFillStyle(1001);
  hmc3->SetLineColor(kBlack);

  totalMC->Scale(scaletodata);

  cout<<scaletodata<<endl;

  TH1F *totalMC_mod = (TH1F*)totalMC->Clone();

  TH1F *h_data = (TH1F*)h1->Clone();
  TH1F *hmc1_mod = (TH1F*)hmc1->Clone();
  TH1F *hmc2_mod = (TH1F*)hmc2->Clone();
  TH1F *hmc3_mod = (TH1F*)hmc3->Clone();

//Divide the bin content with bin width

  for(int ij=1; ij<=hmc1->GetNbinsX(); ++ij){
        double nmc1 = 1.0*hmc1_mod->GetBinContent(ij)/hmc1_mod->GetBinWidth(ij);
        hmc1_mod->SetBinContent(ij,nmc1);

        double nmc2 = 1.0*hmc2_mod->GetBinContent(ij)/hmc2_mod->GetBinWidth(ij);
        hmc2_mod->SetBinContent(ij,nmc2);

        double nmc3 = 1.0*hmc3_mod->GetBinContent(ij)/hmc3_mod->GetBinWidth(ij);
        hmc3_mod->SetBinContent(ij,nmc3);

        double nData = 1.0*h_data->GetBinContent(ij)/h_data->GetBinWidth(ij);
        double BinError = h_data->GetBinError(ij)/h_data->GetBinWidth(ij);
        h_data->SetBinContent(ij,nData);
        h_data->SetBinError(ij,BinError);

        double ntot = 1.0*totalMC_mod->GetBinContent(ij)/totalMC_mod->GetBinWidth(ij);
        double mcBinError = totalMC_mod->GetBinError(ij)/totalMC_mod->GetBinWidth(ij);
        totalMC_mod->SetBinContent(ij,ntot); totalMC_mod->SetBinError(ij,mcBinError);
  }


  hs->Add(hmc1,"hist");
  hs->Add(hmc3,"hist");
  hs->Add(hmc2,"hist");

  hs_mod->Add(hmc1_mod,"hist");
  hs_mod->Add(hmc3_mod,"hist");
  hs_mod->Add(hmc2_mod,"hist");

  double a = h1->GetMaximum();
  hs->Draw("histo");

  h1->SetLineColor(1);
  h1->Draw("pez same");
  h1->SetMarkerSize(0.8);
  h1->SetMarkerStyle(20);


  hs->GetYaxis()->SetLabelSize(0.045);

  hs->GetYaxis()->SetTitle("Events / 0.02");
  hs->GetYaxis()->SetTitleOffset(1.0);
  hs->GetYaxis()->SetTitleSize(0.05);
//  hs->SetMinimum(1.);
//  hs->SetMaximum(800.);
  hs->GetXaxis()->SetLabelSize(0);
  hs->GetXaxis()->SetRangeUser(xmin1,xmax1);


  TLegend *legend1 = new TLegend(0.7, 0.7, 0.86, 0.86);
  legend1->SetTextFont(42);
  legend1->SetLineColor(0);
  legend1->SetTextSize(0.03);
  legend1->SetFillColor(0);
  legend1->AddEntry(h1, "Data", "lep");
  legend1->AddEntry(hmc1, "TTGJets", "f");
  legend1->AddEntry(hmc3, "TTTo2L2Nu", "f");
  legend1->AddEntry(hmc2, "TTToSemiLepton", "f");
  legend1->Draw();

  TLatex *t2a = new TLatex(0.51,0.9," #bf{CMS} #it{Preliminary}             59.7 fb^{-1} (13 TeV, 2018) ");
  t2a->SetNDC();
  t2a->SetTextFont(42);
  t2a->SetTextSize(0.05);
  t2a->SetTextAlign(20);
  t2a->Draw("same");

  c1->cd();

//  TH1F *hist_data = (TH1F*)hdata1->Clone();
  Double_t ratioy[n], ratioEYlow[n], ratioEYhigh[n];

  for(int i = 0; i < n; i++){
      double stackcontent = ((TH1F*)(hs->GetStack()->Last()))->GetBinContent(i+1);
      double stackerror = ((TH1F*)(hs->GetStack()->Last()))->GetBinError(i+1);

      double datacontent = y[i];
      double dataerrorYup = EYhigh[i];
      double dataerrorYdn = EYlow[i];

    if( (stackcontent!=0) && (datacontent !=0) ) {
      ratioy[i] = (datacontent / stackcontent) ;
      ratioEYhigh[i] = ratioy[i]*sqrt(pow((dataerrorYup/datacontent),2) + pow((stackerror/stackcontent),2));
      ratioEYlow[i] = ratioy[i]*sqrt(pow((dataerrorYdn/datacontent),2) + pow((stackerror/stackcontent),2));

    }
  }

  cout << "1" << endl;

  TGraphAsymmErrors *hist_data = new TGraphAsymmErrors(n,x,ratioy,EXlow,EXhigh,ratioEYlow,ratioEYhigh);

  // Ratio Plot

  TPad *pad2 = new TPad("pad2", "newpad",0,0,1,0.3);
  pad2->Draw();
  pad2->cd();
  pad2->SetTopMargin(0.03);
  pad2->SetBottomMargin(0.3);
  pad2->SetRightMargin(0.9);
  pad2->SetFillStyle(0);
  pad2->SetGridx();
  pad2->SetGridy();
  pad2->SetTickx(1);
  pad2->SetTicky(1);
  TLine *line = new TLine(xmin1, 1.,xmax1, 1.);
  line->SetLineColor(kBlack);
  line->SetLineWidth(1.);

  hist_data->SetLineWidth(1.);
  hist_data->SetMarkerStyle(20);
  hist_data->SetMarkerSize(0.8);
  hist_data->SetFillStyle(0);

  TH1F *band = (TH1F*)totalMC->Clone();
  band->Divide(totalMC);
  band->SetFillColor(kGray);
  band->SetLineColor(kWhite);
  band->SetFillStyle(1001);

  cout << "2" << endl;
 
//cout<< totalMC->GetNbinsX() << "  " << h1->GetBinContent(1) << endl;


  TH1F *h_JetEnUp; TH1F *h_JetEnDown; TH1F *h_JetResUp; TH1F *h_JetResDown; TH1F *h_UnclusteredEnUp;TH1F *h_UnclusteredEnDown;
  TGraphAsymmErrors *tottoterree; TGraphAsymmErrors *jererree; TGraphAsymmErrors *erree;

  cout << "3" << endl;
  TGraphAsymmErrors *staterree = (TGraphAsymmErrors*)staterror(h1,totalMC);
  cout << "4" << endl;
  staterree->SetFillColor(kRed-10);
  staterree->SetLineColor(kRed-10);
  staterree->SetFillStyle(1001);
  band->GetXaxis()->SetTitle("BDT");
  band->GetXaxis()->SetLabelSize(0.1);
  band->GetXaxis()->SetTitleSize(0.13);
  band->GetXaxis()->SetRangeUser(xmin1,xmax1);
  band->GetXaxis()->SetTickLength(0.075);
  band->GetYaxis()->SetTitle("Data / MC");
  band->GetYaxis()->SetLabelSize(0.1);
  band->GetYaxis()->SetTitleSize(0.12);
  band->GetYaxis()->SetNdivisions(505);
  band->GetYaxis()->SetTitleOffset(0.4);
  band->GetYaxis()->SetRangeUser(0.0,2.0);
  band->SetTitle("");
  band->SetStats(0);

  band->Draw("");
  staterree->Draw ("e2 same");
  hist_data->Draw("same pez");

  line->Draw("same");
  pad2->RedrawAxis();

  TLegend* legratio(0);
  legratio = new TLegend(0.12,0.35,0.4,0.45);
  legratio->SetFillColor(0);
  legratio->SetBorderSize(0);
  legratio-> SetNColumns(4);
  legratio->SetTextSize(0.07);
  legratio->AddEntry(staterree, "Sim. stat.","f");
//  legratio->Draw();
  c1->SaveAs("BDT_dataMC_test_v1.png");

}

