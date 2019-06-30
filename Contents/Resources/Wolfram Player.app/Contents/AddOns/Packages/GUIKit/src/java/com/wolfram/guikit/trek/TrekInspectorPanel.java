/*
 * @(#)ParameterPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.net.URL;
import java.util.ArrayList;
import java.util.Iterator;

import javax.swing.*;
import javax.swing.border.TitledBorder;

import com.wolfram.guikit.swing.GUIKitImageJLabel;
import com.wolfram.guikit.trek.diva.TrekPane;
import com.wolfram.guikit.trek.event.TrekEvent;
import com.wolfram.guikit.trek.event.TrekListener;

import diva.canvas.Figure;
import diva.canvas.interactor.SelectionEvent;
import diva.canvas.interactor.SelectionListener;
import diva.canvas.interactor.SelectionModel;

/**
 * TrekInspectorPanel
 *
 * @version $Revision: 1.2 $
 */
public class TrekInspectorPanel extends JPanel {

    private static final long serialVersionUID = -1277987975456778940L;
    
	protected TrekPane trekPane = null;
	
	protected JTextField xInitialConditionTextField;
	protected JTextField yInitialConditionTextField;

	protected JTextField originIndependentTextField;
	
	protected JTextField minIndependentVariableTextField;
	protected JTextField maxIndependentVariableTextField;
	
	protected GUIKitImageJLabel originIndependentImageLabel = new GUIKitImageJLabel();
	protected GUIKitImageJLabel xInitialConditionImageLabel = new GUIKitImageJLabel();
	protected GUIKitImageJLabel yInitialConditionImageLabel = new GUIKitImageJLabel();
	
	protected GUIKitImageJLabel minIndependentVariableImageLabel = new GUIKitImageJLabel();
	protected GUIKitImageJLabel maxIndependentVariableImageLabel = new GUIKitImageJLabel();
	
	protected JButton defaultIndependentRangeMinButton;
  protected JButton defaultIndependentRangeMaxButton;
	protected JButton defaultOriginIndependentButton;
  
  protected JPanel independentRangePanel;
  
  public TrekInspectorPanel(TrekPane trekPane) {
    super();
  	this.trekPane = trekPane;
		
		setLayout(new BorderLayout());
    
		JPanel topPanel = new JPanel(new BorderLayout(0,2));
    topPanel.setBorder(new TitledBorder("Conditions"));
 
		JPanel labelPanel = new JPanel(new GridLayout(3,1,4,4));
		JPanel fieldsPanel = new JPanel(new GridLayout(3,1,4,4));
		
		labelPanel.add(xInitialConditionImageLabel);
		xInitialConditionTextField = new JTextField("");
		xInitialConditionTextField.setColumns(8);
		xInitialConditionTextField.setHorizontalAlignment(JTextField.RIGHT);
    // Currently not used because we use Mathematica kernel eval for parsing text input
    /*
		xInitialConditionTextField.addActionListener(
		  new ActionListener() {
	      public void actionPerformed(ActionEvent e) {updateSelectionInitialConditions(); };
			  }
		   );
    */
		xInitialConditionTextField.setEnabled(false);
		fieldsPanel.add(xInitialConditionTextField);
		
		labelPanel.add(yInitialConditionImageLabel);
		yInitialConditionTextField = new JTextField("");
		yInitialConditionTextField.setColumns(8);
		yInitialConditionTextField.setHorizontalAlignment(JTextField.RIGHT);
    // Currently not used because we use Mathematica kernel eval for parsing text input
    /*
		yInitialConditionTextField.addActionListener(
		  new ActionListener() {
		    public void actionPerformed(ActionEvent e) {updateSelectionInitialConditions(); };
	      }
		  );
    */
		yInitialConditionTextField.setEnabled(false);
		fieldsPanel.add(yInitialConditionTextField);
				
		labelPanel.add(originIndependentImageLabel);
    
    JPanel independentFieldPanel = new JPanel(new BorderLayout());
		originIndependentTextField = new JTextField("");
		originIndependentTextField.setColumns(8);
		originIndependentTextField.setHorizontalAlignment(JTextField.RIGHT);
		// Currently not used because we use Mathematica kernel eval for parsing text input
		/*
		defaultOriginIndependentButton.addActionListener(
			new ActionListener() {
				public void actionPerformed(ActionEvent e) {setDefaultOriginIndependent(); };
				}
			);
		*/
    independentFieldPanel.add(originIndependentTextField, BorderLayout.CENTER);
    ImageIcon ic = null;
    URL imageURL = TrekInspectorPanel.class.getClassLoader().getResource("images/trek/apply.gif");
    if (imageURL != null) {
      ic = new ImageIcon(imageURL);
      }
    defaultOriginIndependentButton = new JButton(ic);
    Dimension sze = new Dimension(20,20);
    defaultOriginIndependentButton.setPreferredSize(sze);
    defaultOriginIndependentButton.setMinimumSize(sze);
    defaultOriginIndependentButton.setMaximumSize(sze);
    defaultOriginIndependentButton.setToolTipText("Use this value when creating new treks");
    independentFieldPanel.add(defaultOriginIndependentButton, BorderLayout.EAST);
    
		fieldsPanel.add(independentFieldPanel);
    
		topPanel.add(labelPanel, BorderLayout.WEST);
		topPanel.add(fieldsPanel, BorderLayout.CENTER);

		defaultIndependentRangeMinButton = new JButton(ic);
    defaultIndependentRangeMinButton.setToolTipText("Use this range min value when creating new treks");
    defaultIndependentRangeMinButton.setPreferredSize(sze);
    defaultIndependentRangeMinButton.setMinimumSize(sze);
    defaultIndependentRangeMinButton.setMaximumSize(sze);
    // Currently not used because we use Mathematica kernel eval for parsing text input
    /*
		defaultIndependentRangeButton.addActionListener(
			new ActionListener() {
				public void actionPerformed(ActionEvent e) {setDefaultIndependentRange(); };
				}
			);
    */
		
    independentRangePanel = new JPanel(new BorderLayout(0,2));
    independentRangePanel.setBorder(new TitledBorder("Independent Range"));
		labelPanel = new JPanel(new GridLayout(2,1,4,4));
		fieldsPanel = new JPanel(new GridLayout(2,1,4,4));
		
		labelPanel.add(minIndependentVariableImageLabel);
		minIndependentVariableTextField = new JTextField("");
		minIndependentVariableTextField.setColumns(8);
		minIndependentVariableTextField.setHorizontalAlignment(JTextField.RIGHT);
    // Currently not used because we use Mathematica kernel eval for parsing text input
    /*
		minIndependentVariableTextField.addActionListener(
		  new ActionListener() {
		    public void actionPerformed(ActionEvent e) {updateSelectionIndependentRange(); };
				}
			);
    */
    JPanel independentMinPanel = new JPanel(new BorderLayout());
    independentMinPanel.add(minIndependentVariableTextField, BorderLayout.CENTER);
    independentMinPanel.add(defaultIndependentRangeMinButton, BorderLayout.EAST);
		fieldsPanel.add(independentMinPanel);

		labelPanel.add(maxIndependentVariableImageLabel);
    
    defaultIndependentRangeMaxButton = new JButton(ic);
    defaultIndependentRangeMaxButton.setToolTipText("Use this range max value when creating new treks");
    defaultIndependentRangeMaxButton.setPreferredSize(sze);
    defaultIndependentRangeMaxButton.setMinimumSize(sze);
    defaultIndependentRangeMaxButton.setMaximumSize(sze);
    
		maxIndependentVariableTextField = new JTextField("");
		maxIndependentVariableTextField.setColumns(8);
		maxIndependentVariableTextField.setHorizontalAlignment(JTextField.RIGHT);
    // Currently not used because we use Mathematica kernel eval for parsing text input
    /*
		maxIndependentVariableTextField.addActionListener(
		  new ActionListener() {
		    public void actionPerformed(ActionEvent e) {updateSelectionIndependentRange(); };
		    }
	    );
    */
    JPanel independentMaxPanel = new JPanel(new BorderLayout());
    independentMaxPanel.add(maxIndependentVariableTextField, BorderLayout.CENTER);
    independentMaxPanel.add(defaultIndependentRangeMaxButton, BorderLayout.EAST);
    
		fieldsPanel.add(independentMaxPanel);
		independentRangePanel.add(labelPanel, BorderLayout.WEST);
		independentRangePanel.add(fieldsPanel, BorderLayout.CENTER);
				 
		add(topPanel, BorderLayout.NORTH);
		add(independentRangePanel, BorderLayout.SOUTH);
		
		SelectionModel model = trekPane.getTrekController().getSelectionModel();
		model.addSelectionListener( new SelectionListener() {
			public void selectionChanged(SelectionEvent e) {
				updateFromSelection();
				}
		 	});
		 	
		trekPane.addTrekListener(new TrekListener() {
		   public void trekOriginDidChange(TrekEvent e) {
		   	 if(xInitialConditionTextField.isEnabled()) 
				   updateFromSelection();
		     }
		   public void trekIndependentRangeDidChange(TrekEvent e) {}
			 });
				
		updateFromSelection();
    }

  public void setSelectionColor(Color newColor) {
    trekPane.setSelectionColor(newColor);
    }
  
  public void setupForNoIndependentVariable() {
    independentRangePanel.getParent().remove(independentRangePanel);
    }
    
  public void setupForFirstOrder() {
		originIndependentTextField.getParent().remove(originIndependentTextField);
		originIndependentImageLabel.getParent().remove(originIndependentImageLabel);
		defaultOriginIndependentButton.getParent().remove(defaultOriginIndependentButton);
  	}
  
  public void updateFromSelection() {
    SelectionModel sm = trekPane.getTrekController().getSelectionModel();             
    ArrayList selKeys = new ArrayList();
    TrekToolBar toolBar = trekPane.getTrekToolBar();
    JTextField colorWell = null;
    JButton selectionDisplayModeButton = null;
    
    if (toolBar != null) {
      colorWell = toolBar.getColorWell();
      selectionDisplayModeButton = toolBar.getSelectionDisplayModeButton();
      }
    
		for(Iterator iter = sm.getSelection(); iter.hasNext();){
			String key = trekPane.getTrekIdFromTarget((Figure)iter.next());
			if (key != null) {
				selKeys.add(key);
				}
			}
	
	  int count = selKeys.size();
	  if (count > 1) {
			double[] rnge = trekPane.getDefaultIndependentRange();
			double originIndep = trekPane.getDefaultOriginIndependent();
			minIndependentVariableTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[0]));
			maxIndependentVariableTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[1]));
			originIndependentTextField.setText(TrekPane.defaultDecimalFormat.format(originIndep));
			xInitialConditionTextField.setText("<multiple selection>");
			xInitialConditionTextField.setEnabled(false);
			yInitialConditionTextField.setText("<multiple selection>");
			yInitialConditionTextField.setEnabled(false);
      
      if (selectionDisplayModeButton != null) selectionDisplayModeButton.setEnabled(true);
      if (toolBar != null)
        toolBar.setSelectionDisplayMode(trekPane.getTrekFigure((String)selKeys.get(0)).getDisplayMode());
      if (colorWell != null) {
        colorWell.setEnabled(true);
        colorWell.setBackground(trekPane.getTrekFigure((String)selKeys.get(0)).getColor());
        }
			return;
	    }
	  else if (count == 1) {
      String id = (String)selKeys.get(0);
			Trek t = trekPane.getTrek(id);
			if (t != null) {
				double[] rnge = t.getIndependentRange();
				double originIndep = t.getOriginIndependent();
				minIndependentVariableTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[0]));
				maxIndependentVariableTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[1]));
				rnge = t.getOrigin();
				originIndependentTextField.setText(TrekPane.defaultDecimalFormat.format(originIndep));
				xInitialConditionTextField.setEnabled(true);
				yInitialConditionTextField.setEnabled(true);
				xInitialConditionTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[0]));
				yInitialConditionTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[1]));
        if (selectionDisplayModeButton != null) selectionDisplayModeButton.setEnabled(true);
        if (toolBar != null)
          toolBar.setSelectionDisplayMode(trekPane.getTrekFigure(id).getDisplayMode());
        if (colorWell != null) {
          colorWell.setEnabled(true);
          colorWell.setBackground( trekPane.getTrekFigure(id).getColor());
          }
				return;
				}
	    }
			
		double[] rnge = trekPane.getDefaultIndependentRange();
		double originIndep = trekPane.getDefaultOriginIndependent();
		minIndependentVariableTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[0]));
		maxIndependentVariableTextField.setText(TrekPane.defaultDecimalFormat.format(rnge[1]));
		originIndependentTextField.setText(TrekPane.defaultDecimalFormat.format(originIndep));
		xInitialConditionTextField.setText("<no selection>");
		xInitialConditionTextField.setEnabled(false);
		yInitialConditionTextField.setText("<no selection>");
		yInitialConditionTextField.setEnabled(false);
    if (selectionDisplayModeButton != null) selectionDisplayModeButton.setEnabled(false);
    if (colorWell != null) {
      colorWell.setEnabled(false);
      colorWell.setBackground(Color.WHITE);
      }
  	}
  
  // Currently not used because we use Mathematica kernel eval for parsing text input
  public void updateSelectionInitialConditions() {
  	double[] newVals = new double[]{0.0,0.0};
		try {
  		newVals[0] = Double.parseDouble(xInitialConditionTextField.getText());
			newVals[1] = Double.parseDouble(yInitialConditionTextField.getText());
  		trekPane.setSelectionInitialConditions(newVals);
  		trekPane.getCanvas().requestFocus();
			}
		catch (NumberFormatException ne) {
			}
  	}
	// Currently not used because we use Mathematica kernel eval for parsing text input
	public void updateOriginIndependent() {
		double newVal = 0.0;
		try {
			newVal = Double.parseDouble(originIndependentTextField.getText());
			trekPane.setSelectionOriginIndependent(newVal);
			trekPane.getCanvas().requestFocus();
			}
		catch (NumberFormatException ne) {
			}
		}
	// Currently not used because we use Mathematica kernel eval for parsing text input
	public void setDefaultOriginIndependent() {
		double newVal = 0.0;
		try {
			newVal = Double.parseDouble(originIndependentTextField.getText());
			trekPane.setDefaultOriginIndependent(newVal);
			trekPane.getCanvas().requestFocus();
			}
		catch (NumberFormatException ne) {
			}
		}
  // Currently not used because we use Mathematica kernel eval for parsing text input
	public void updateSelectionIndependentRange() {
		double[] newVals = new double[]{0.0,0.0};
		try {
			newVals[0] = Double.parseDouble(minIndependentVariableTextField.getText());
			newVals[1] = Double.parseDouble(maxIndependentVariableTextField.getText());
			trekPane.setSelectionIndependentRange(newVals);
			trekPane.getCanvas().requestFocus();
			}
		catch (NumberFormatException ne) {
			}
		}
  // Currently not used because we use Mathematica kernel eval for parsing text input
	public void setDefaultIndependentRange() {
		double[] newVals = new double[]{0.0,0.0};
		try {
		  newVals[0] = Double.parseDouble(minIndependentVariableTextField.getText());
		  newVals[1] = Double.parseDouble(maxIndependentVariableTextField.getText());
		  trekPane.setDefaultIndependentRange(newVals);
			trekPane.getCanvas().requestFocus();
			}
		catch (NumberFormatException ne) {
			}
		}

  
  public JButton getDefaultIndependentRangeMinButton() {return defaultIndependentRangeMinButton;}
  public JButton getDefaultIndependentRangeMaxButton() {return defaultIndependentRangeMaxButton;}
	public JButton getDefaultOriginIndependentButton() {return defaultOriginIndependentButton;}
	public JTextField getOriginIndependentTextField() {return originIndependentTextField;}
  public JTextField getXInitialConditionTextField() {return xInitialConditionTextField;}
  public JTextField getYInitialConditionTextField() {return yInitialConditionTextField;}
  public JTextField getMinIndependentVariableTextField() {return minIndependentVariableTextField;}
  public JTextField getMaxIndependentVariableTextField() {return maxIndependentVariableTextField;}
  
	public GUIKitImageJLabel getOriginIndependentImageLabel() {return originIndependentImageLabel;}
	public GUIKitImageJLabel getXInitialConditionImageLabel() {return xInitialConditionImageLabel;}
	public GUIKitImageJLabel getYInitialConditionImageLabel() {return yInitialConditionImageLabel;}
  public GUIKitImageJLabel getMinIndependentVariableImageLabel() {return minIndependentVariableImageLabel;}
	public GUIKitImageJLabel getMaxIndependentVariableImageLabel() {return maxIndependentVariableImageLabel;}
	
}