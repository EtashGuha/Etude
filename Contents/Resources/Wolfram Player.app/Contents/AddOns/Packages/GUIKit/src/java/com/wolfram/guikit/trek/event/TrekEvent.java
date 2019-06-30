/*
 * @(#)ParameterEvent.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.event;

import java.awt.AWTEvent;

/**
 * ParameterEvent
 */
public class TrekEvent extends AWTEvent  {

  /** The offset from the reserved event ids that the ids for this event start at. */
  private static final int ID_OFFSET = 52800;

  /** Marks the first integer id for the range of paint event ids. */
  public static final int TREK_EVENT_FIRST = RESERVED_ID_MAX + ID_OFFSET;

  /**
      The parameter changed event type.  This event is delivered after
      a parameter's value has changed.
  */
  public static final int ORIGIN_DID_CHANGE	= TREK_EVENT_FIRST;
	public static final int INDEPENDENT_RANGE_DID_CHANGE	= TREK_EVENT_FIRST + 1;
	
  /**
      Marks the last integer id for the range of paint event ids.
  */
  public static final int TREK_EVENT_LAST  = INDEPENDENT_RANGE_DID_CHANGE;

  /* JDK 1.1 serialVersionUID */
  private static final long serialVersionUID = -1287987975456788948L;
  
  private String key = null;
  private double[] origin;
  private double originIndependent;
	private double[] independentRange;
	
  /**
	 * @param source the component that is being painted
	 * @param g the graphics context for the painting
   * @param id the unique identifier for the type of paint event
	 */
  public TrekEvent(String key, double[] origin, double originIndependent, double[] range, int id) {
    super(key, id);
    this.key = key;
    this.origin = origin;
    this.originIndependent = originIndependent;
		this.independentRange = range;
    }

  public String getKey() {return key;}
  public double[] getOrigin() {return origin;}
	public double getOriginIndependent() {return originIndependent;}
	public double[] getIndependentRange() {return independentRange;}
	
  }
