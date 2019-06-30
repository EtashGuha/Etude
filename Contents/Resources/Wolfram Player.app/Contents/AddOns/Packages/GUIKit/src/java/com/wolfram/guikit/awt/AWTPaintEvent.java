/*
 * @(#)AWTPaintEvent.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import java.awt.AWTEvent;
import java.awt.Graphics;

/**
 * AWTPaintEvent is the event type that is sent to AWTPaintListeners to notify them
 * when painting is occurring on a component and the event contains an active
 * Graphics context that the listeners could choose to draw within.
 */
public class AWTPaintEvent extends AWTEvent  {

  /** The offset from the reserved event ids that the ids for this event start at. */
  private static final int ID_OFFSET = 52000;

  /** Marks the first integer id for the range of paint event ids. */
  public static final int AWTPAINT_EVENT_FIRST = RESERVED_ID_MAX + ID_OFFSET;

  /**
      The will paint event type.  This event is delivered before
      the component paints, even before super.paint is called.
  */
  public static final int WILL_PAINT	= AWTPAINT_EVENT_FIRST;

  /**
      The did paint event type.  This event is delivered after
      the component and its super paints.
  */
  public static final int DID_PAINT = 1 + AWTPAINT_EVENT_FIRST;

  /**
      The will paint event type.  This event is delivered before
      the component gets the update call
  */
  public static final int WILL_UPDATE	= 2 + AWTPAINT_EVENT_FIRST;

  /**
      The did paint event type.  This event is delivered after
      the component received the update call.
  */
  public static final int DID_UPDATE = 3 + AWTPAINT_EVENT_FIRST;

  /**
      Marks the last integer id for the range of paint event ids.
  */
  public static final int AWTPAINT_EVENT_LAST  = DID_UPDATE;

  /* JDK 1.1 serialVersionUID */
  private static final long serialVersionUID = -1287987973456298748L;

  private Graphics graphics;

  /**
	 * @param source the component that is being painted
	 * @param g the graphics context for the painting
   * @param id the unique identifier for the type of paint event
	 */
  public AWTPaintEvent(Object source, Graphics g, int id) {
    super(source, id);
    this.graphics = g;
    }

	/**
	 * Gives the active graphics context used in painting.
	 *
	 * @return the active graphics context instance
	 */
  public Graphics getGraphics() {return graphics;}

  }
