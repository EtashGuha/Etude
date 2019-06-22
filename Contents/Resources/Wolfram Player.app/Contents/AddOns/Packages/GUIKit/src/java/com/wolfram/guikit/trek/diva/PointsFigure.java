/*
 * @(#)AxesFigure.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.diva;

import java.awt.Graphics2D;
import java.awt.Paint;

import diva.canvas.toolbox.PathFigure;
import diva.util.java2d.Polyline2D;

import java.awt.geom.Rectangle2D;
import java.awt.geom.RectangularShape;

/**
 * PointsFigure draws a decorator shape at the points instead
 * of a continuous line, but consider making this the default
 * trek figure with a mode setting for display
 *
 * @version $Revision: 1.1 $
 */
public class PointsFigure extends PathFigure {

  protected int displayMode = TrekFigure.LINE;
  
  //protected RectangularShape pointsShape = new Ellipse2D.Double(-0.5, -0.5, 1, 1);
  // For now focus on simple pixel rendering instead of Ellipse
  protected RectangularShape pointsShape = new Rectangle2D.Double(-0.5, -0.5, 1, 1);
  
  /** Create a new figure with the given shape. The figure, by
   *  default, is stroked with a unit-width continuous black stroke.
   */
  public PointsFigure(Polyline2D shape) {
    super(shape);
    }

  /** Create a new figure with the given shape and width.
   * The default paint is black.
   */
  public PointsFigure(Polyline2D shape, float lineWidth) {
    super(shape, lineWidth);
  }

  /** Create a new figure with the given paint and width.
   */
  public PointsFigure(Polyline2D shape, Paint paint, float lineWidth) {
    super(shape, paint, lineWidth);
  }
  
  public RectangularShape getPointsShape() {return pointsShape;}
  public void setPointsShape(RectangularShape s) {
    pointsShape = s;
    repaint();
    }
  public int getDisplayMode() {return displayMode;}
  public void setDisplayMode(int newMode) {
    int oldMode = displayMode;
    switch (newMode) {
      case TrekFigure.POINTS: displayMode = TrekFigure.POINTS; break;
      default: displayMode = TrekFigure.LINE; break;
      }
    if (oldMode != displayMode) 
      repaint();
    }
    
  /** Paint the figure. The figure is redrawn with the current
    *  shape, fill, and outline.
    */
  public void paint(Graphics2D g) {
    if (!isVisible()) return;
    
    Polyline2D s = (Polyline2D)getShape();
    if (s == null) return;
    
    g.setStroke(getStroke());
    g.setPaint(getStrokePaint());
      
    if (displayMode == TrekFigure.POINTS && pointsShape != null) {
      int count = s.getVertexCount();
      for (int i = 0; i < count; ++i) {
        pointsShape.setFrame(s.getX(i)-0.5, s.getY(i)-0.5, 1.0, 1.0);
        g.draw(pointsShape);
        }
      }
    else {
      g.draw(s);
      }

    }

}
