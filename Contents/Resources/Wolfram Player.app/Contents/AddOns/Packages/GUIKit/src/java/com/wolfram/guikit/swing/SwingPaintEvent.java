/*
 * @(#)SwingPaintEvent.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.AWTEvent;
import java.awt.Graphics;

/**
 * SwingPaintEvent is the event type that is sent to SwingPaintListeners to notify them
 * when painting is occurring on a component, its children and its border.
 * The event contains an active Graphics context that the listeners could choose to draw within.
 */
public class SwingPaintEvent extends AWTEvent  {

  /** The offset from the reserved event ids that the ids for this event start at. */
  private static final int ID_OFFSET = 52100;

  /** Marks the first integer id for the range of paint event ids. */
  public static final int SWINGPAINT_EVENT_FIRST = RESERVED_ID_MAX + ID_OFFSET;

  /**
      The will paint event type.  This event is delivered before
      the component paints itself.
  */
  public static final int WILL_PAINT_COMPONENT	= SWINGPAINT_EVENT_FIRST;

  /**
      The did paint event type.  This event is delivered after
      the component paints itself.
  */
  public static final int DID_PAINT_COMPONENT = 1 + SWINGPAINT_EVENT_FIRST;

  /**
      The will paint event type.  This event is delivered before
      the component paints its border.
  */
  public static final int WILL_PAINT_BORDER	= 2 + SWINGPAINT_EVENT_FIRST;

  /**
      The did paint event type.  This event is delivered after
      the component paints its border.
  */
  public static final int DID_PAINT_BORDER = 3 + SWINGPAINT_EVENT_FIRST;

  /**
      The will paint event type.  This event is delivered before
      the component paints its children components.
  */
  public static final int WILL_PAINT_CHILDREN	= 4 + SWINGPAINT_EVENT_FIRST;

  /**
      The did paint event type.  This event is delivered after
      the component paints its children components.
  */
  public static final int DID_PAINT_CHILDREN = 5 + SWINGPAINT_EVENT_FIRST;

  /**
      Marks the last integer id for the range of paint event ids.
  */
  public static final int SWINGPAINT_EVENT_LAST  = DID_PAINT_CHILDREN;

  /* JDK 1.1 serialVersionUID */
  private static final long serialVersionUID = -1287987973456298749L;

  private Graphics graphics;

  /**
	 * @param source the component that is being painted
	 * @param g the graphics context for the painting
   * @param id the unique identifier for the type of paint event
	 */
  public SwingPaintEvent(Object source, Graphics g, int id) {
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
