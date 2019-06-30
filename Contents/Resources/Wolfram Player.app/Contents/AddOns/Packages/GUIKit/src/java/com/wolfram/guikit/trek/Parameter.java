/*
 * @(#)Parameter.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek;

import com.wolfram.guikit.trek.event.ParameterEventHandler;
import com.wolfram.guikit.trek.event.ParameterListener;

import com.wolfram.jlink.Expr;

/**
 * Parameter
 *
 * @version $Revision: 1.1 $
 */
public class Parameter {

  protected Number value = null;
  
  protected Number defaultValue;
  protected Number minValue;
  protected Number maxValue;

  protected String description;
  protected String key;
  
  protected Expr name;
  
  protected ParameterEventHandler eventHandler;
  
  public Parameter() {
    this(null, "", new Double(0.0), new Double(0.0), new Double(1.0));
    }
  
	public Parameter(String key, String description, Number defaultValue, Number minValue, Number maxValue) {
    setKey(key);
    setDescription(description);
    setDefaultValue(defaultValue);
    setMinValue(minValue);
    setMaxValue(maxValue);
	  }

  public void addParameterListener(ParameterListener l) {
    if (l != null) {
      if (eventHandler == null) eventHandler = new ParameterEventHandler();
      eventHandler.addParameterListener(l);
      }
    }
  
  public void removeParameterListener(ParameterListener l) {
    if (l != null) {
      if (eventHandler != null)
        eventHandler.removeParameterListener(l);
      }
    }
  
  public Expr getName() {return name;}
  public void setName(Expr name) {
   this.name = name;
   }
  
  public String getKey() {return key;}
  public void setKey(String key) {
    this.key = key;
    }
  
  public String getDescription() {return description;}
  public void setDescription(String description) {
    this.description = description;
    }
  
  public Number getDefaultValue() {return defaultValue;}
  public void setDefaultValue(Number defaultValue) {
    this.defaultValue = defaultValue;
    value = defaultValue;
    }
    
  public Number getValue() {return value; }
  public void setValue(Number value) {
    setValue(value, false);
    }
    
  public void setValue(Number value, boolean valueIsAdjusting) {
    Number oldValue = this.value;
    this.value = value;
    if (eventHandler != null && value != null && !value.equals(oldValue)) {
      eventHandler.fireDidChange(value, oldValue, this, valueIsAdjusting);
      }
    }

  public Number getMinValue() {return minValue; }
  public void setMinValue(Number minValue) {
    this.minValue = minValue;
    if ( value.doubleValue() < minValue.doubleValue()) setValue(minValue);
    }
  
  public Number getMaxValue() {return maxValue; }
  public void setMaxValue(Number maxValue) {
    this.maxValue = maxValue;
    if (value.doubleValue() > maxValue.doubleValue()) setValue(maxValue);
    }
    
}