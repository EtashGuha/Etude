/*
 * @(#)ParameterUI.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.wolfram.guikit.swing.GUIKitImageJLabel;
import com.wolfram.guikit.trek.diva.TrekPane;
import com.wolfram.guikit.trek.event.ParameterEvent;
import com.wolfram.guikit.trek.event.ParameterListener;

import java.awt.BorderLayout;
import java.awt.Dimension;

/**
 * ParameterUI
 *
 * @version $Revision: 1.2 $
 */
public class ParameterUI extends JPanel implements ParameterListener {
  
  private static final long serialVersionUID = -1287987925456728248L;
    
  protected JTextField text;
  protected JSlider control;
  protected JLabel label = new GUIKitImageJLabel();
  protected Parameter param;
  protected TrekPane trekPane = null;
  
	public ParameterUI() {
    super();
    }

  public JLabel getLabel() {return label;}
  public void setLabel(JLabel l) {
    label = l;
    }
  
  public JTextField getTextField() {return text;}
    
  public TrekPane getTrekPane() {return trekPane;}
  public void setTrekPane(TrekPane t) {
    trekPane = t;
    }
    
  public void setParameter(Parameter p) {
    param = p;
    param.addParameterListener(this);
    
    int minHeight = 0;
    
    setLayout( new BorderLayout());
    
    JPanel topPanel = new JPanel();
    topPanel.setLayout(new BorderLayout());
    topPanel.add(label, BorderLayout.WEST);
    text = new JTextField(param.getDefaultValue().toString(), 16);
    text.setHorizontalAlignment(JTextField.RIGHT);
    
    if (text.getMinimumSize() != null)
      minHeight += text.getMinimumSize().getHeight();
      
    // Currently not used because we use Mathematica kernel eval for parsing text input
    /*
    text.addActionListener( new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        Double val = new Double(0.0);
        try {
          val = new Double(text.getText());
          }
        catch (Exception ex){}
        setValue(val);
        if (trekPane != null)
          trekPane.getCanvas().requestFocus();
        }
      });
    */
    
    topPanel.add(text, BorderLayout.CENTER);

    
    control = new JSlider(0, 100, toAdjustmentValue(param.getDefaultValue()));
    
    if (control.getMinimumSize() != null)
      minHeight += control.getMinimumSize().getHeight();
      
    minHeight = Math.max(minHeight, 45);
    
    add(topPanel, BorderLayout.NORTH);
    add(control, BorderLayout.CENTER);
    
    setMinimumSize(new Dimension(50, minHeight));
    setMaximumSize(new Dimension(2000, minHeight));
    setPreferredSize(new Dimension(75, minHeight));
    
    control.addChangeListener( 
      new ChangeListener() {
        public void stateChanged(ChangeEvent e) {
          Number currentValue = fromAdjustmentValue(control.getValue());
          param.setValue(currentValue, control.getValueIsAdjusting());
          if (!control.getValueIsAdjusting()) {
            if (trekPane != null)
              trekPane.getCanvas().requestFocus();
            }
          }
        }
      );
    }
  
  public JSlider getControl() {return control;}

  private Number fromAdjustmentValue(int adjVal) {
    if (param == null) return null;
    return new Double( param.getMinValue().doubleValue() +
      (param.getMaxValue().doubleValue() - param.getMinValue().doubleValue()) * (double)adjVal/100.0);
    }

  private int toAdjustmentValue(Number val) {
    if (param == null) return -1;
    return (int) Math.round( 100.0 *(val.doubleValue() - param.getMinValue().doubleValue()) /
         (param.getMaxValue().doubleValue() - param.getMinValue().doubleValue()));
    }

  public Number getValue() {
    return fromAdjustmentValue( control.getValue());
    }

  public void setValue(Number value) {
    Number useValue = value;
    if (param == null || value == null) return;
    if (value.doubleValue() < param.getMinValue().doubleValue())
      useValue = param.getMinValue();
    else if (value.doubleValue() > param.getMaxValue().doubleValue())
      useValue = param.getMaxValue();
    control.setValue( toAdjustmentValue(useValue) );
    text.setText( TrekPane.defaultDecimalFormat.format(useValue.doubleValue()));
    }

  public void didChange(ParameterEvent e) {
    setValue(e.getNewValue());
    }

}