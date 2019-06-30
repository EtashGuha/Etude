/*
 * @(#)ParameterEvent.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.event;

import java.awt.AWTEvent;

import com.wolfram.guikit.trek.Parameter;

/**
 * ParameterEvent
 */
public class ParameterEvent extends AWTEvent  {

  /** The offset from the reserved event ids that the ids for this event start at. */
  private static final int ID_OFFSET = 52700;

  /** Marks the first integer id for the range of paint event ids. */
  public static final int PARAMETER_EVENT_FIRST = RESERVED_ID_MAX + ID_OFFSET;

  /**
      The parameter changed event type.  This event is delivered after
      a parameter's value has changed.
  */
  public static final int DID_CHANGE	= PARAMETER_EVENT_FIRST;

  /**
      Marks the last integer id for the range of paint event ids.
  */
  public static final int PARAMETER_EVENT_LAST  = DID_CHANGE;

  /* JDK 1.1 serialVersionUID */
  private static final long serialVersionUID = -1287987973456788748L;

  protected Number newValue;
  protected Number oldValue;
  protected boolean valueIsAdjusting = false;
  
 // public ParameterEvent(Parameter source, Number newValue, Number oldValue, int id) {
 //   this(source, newValue, oldValue, id, false);
//    }
  /**
	 * @param source the component that is being painted
	 * @param g the graphics context for the painting
   * @param id the unique identifier for the type of paint event
	 */
  public ParameterEvent(Parameter source, Number newValue, Number oldValue, int id, boolean valueIsAdjusting) {
    super(source, id);
    this.newValue = newValue;
    this.oldValue = oldValue;
    this.valueIsAdjusting = valueIsAdjusting;
    }

  public Number getNewValue() {return newValue;}
  public Number getOldValue() {return oldValue;}
  
  public boolean getValueIsAdjusting() {return valueIsAdjusting;}
   
  }
