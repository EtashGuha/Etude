/*
 * @(#)TrekFigure.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.diva;

import java.awt.Color;

import java.awt.geom.Ellipse2D;
import java.util.Random;

import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;
import com.wolfram.guikit.diva.ArrowheadFigure;
import com.wolfram.jlink.Expr;

import diva.util.java2d.Polyline2D;
import diva.canvas.CompositeFigure;
import diva.canvas.Figure;

import diva.canvas.toolbox.BasicFigure;

/**
 * TrekFigure is a diva Figure subclass that represents a drawn trek.
 *
 * @version $Revision: 1.4 $
 */
public class TrekFigure extends CompositeFigure {

  private static Random randomColorSeed = new Random();
  
  public static final int LINE = 1;
  public static final int POINTS = 2;
  
  protected int displayMode = TrekFigure.LINE;
  
  private static final double ELLIPSE_RADIUS = 3.0;
  
  private BasicFigure selectionFigure;
  private PointsFigure trekLine;
  private ArrowheadFigure arrowhead;
  
  private Color pathStrokeColor = Color.blue;
  private Color highlightFillColor = Color.black;
  
  private double[] origin;
  private double points[][];
  
  public TrekFigure() {
    this(null);
    }
	public TrekFigure(String key) {
		super();
    setUserObject(key);
	  }
 
  protected Polyline2D createLineShape(double xorigin, double xscale, double yorigin, double yscale) {
    Polyline2D p = null;
    if (points[0].length > 0) {
      p = new Polyline2D.Double(points[0].length);
      p.moveTo( (points[0][0]- xorigin)*xscale, (yorigin-points[1][0])*yscale);
      for (int i = 1; i < points[0].length; ++i)
        p.lineTo( (points[0][i]- xorigin)*xscale, (yorigin-points[1][i])*yscale);
      }
    return p;
    }
  
  protected Ellipse2D.Double createSelectionShape(double xorigin, double xscale, double yorigin, double yscale) {
    return new Ellipse2D.Double(
      (origin[0] - xorigin)*xscale - ELLIPSE_RADIUS, 
      (yorigin - origin[1])*yscale - ELLIPSE_RADIUS,
          ELLIPSE_RADIUS*2, ELLIPSE_RADIUS*2);
    }
  
  protected void updateArrowhead(double xorigin, double xscale, double yorigin, double yscale) {
    if (points[0].length >= 2) {
      if (displayMode != POINTS) {
        arrowhead.setVisible(true);
        }
      double beginX = (points[0][points[0].length-2]- xorigin)*xscale;
      double endX = (points[0][points[0].length-1]- xorigin)*xscale;
      double beginY = (yorigin-points[1][points[0].length-2])*yscale;
      double endY = (yorigin-points[1][points[0].length-1])*yscale;
      arrowhead.setOrigin(endX, endY);
      arrowhead.setNormal(Math.atan2(beginY-endY, beginX-endX));
      }
    else arrowhead.setVisible(false);
    }
  
  public void update(double[] origin, double points[][],
       double xorigin, double xscale, double yorigin, double yscale) {
    this.origin = origin;
    this.points = points;
    recreate(xorigin, xscale, yorigin, yscale);
    }
    
  public void recreate(double xorigin, double xscale, double yorigin, double yscale) {
    trekLine.setShape( createLineShape(xorigin, xscale, yorigin, yscale));
    selectionFigure.setShape( createSelectionShape(xorigin, xscale, yorigin, yscale));
    updateArrowhead(xorigin, xscale, yorigin, yscale);
    }
  
  public int getDisplayMode() {return displayMode;}
  public void setDisplayMode(int newMode) {
    int oldMode = displayMode;
    switch (newMode) {
      case TrekFigure.POINTS: 
        displayMode = TrekFigure.POINTS;
        if (trekLine != null) trekLine.setDisplayMode(newMode);
        if (arrowhead != null) arrowhead.setVisible(false);
        break;
      default: 
        displayMode = TrekFigure.LINE; 
        if (trekLine != null) trekLine.setDisplayMode(newMode);
        if (arrowhead != null) arrowhead.setVisible(true);
        break;
      }
    if (oldMode != displayMode) 
      repaint();
    }
    
  public void init(double[] origin, double points[][],
       double xorigin, double xscale, double yorigin, double yscale) {
    this.origin = origin;
    this.points = points;

    trekLine = new PointsFigure( createLineShape(xorigin, xscale, yorigin, yscale));
    trekLine.setLineWidth(0.5f);
    trekLine.setStrokePaint( getPathStrokeColor());
    trekLine.setUserObject(getKey());
    add(trekLine);
      
    selectionFigure = new BasicFigure(
      createSelectionShape(xorigin, xscale, yorigin, yscale), 
      getOriginFillColor());
    selectionFigure.setUserObject(getKey());
    add(selectionFigure);
    
    arrowhead = new ArrowheadFigure();
    add(arrowhead);
    updateArrowhead(xorigin, xscale, yorigin, yscale);
    }

  public Expr getColorExpr() {
    Expr e = null;
    try {
      e = (Expr)MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
      Color.class, getColor(), Expr.class);
      }
    catch (MathematicaBSFException ex) {}
    return e;
    }
  public void setColorExpr(Expr c) {
    Color col = null;
    if (c != null) {
      try {
        col = (Color)MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
          Expr.class, c, Color.class);
        }
      catch (MathematicaBSFException ex) {}
      }
    setColor(col);
    }
    
  public Color createRandomColor() {
    return Color.getHSBColor(randomColorSeed.nextFloat(), (float)1.0, (float)1.0);
    }
  
  public Color getColor() {return getPathStrokeColor();}
  public void setColor(Color c) {
    Color useColor = c;
    if (useColor == null) {
      useColor = createRandomColor();
      }
    setPathStrokeColor(useColor);
    setOriginFillColor(useColor);
    }
    
  public Color getPathStrokeColor() {
    return pathStrokeColor;
    }
  public void setPathStrokeColor(Color c) {
    pathStrokeColor = c;
    if (trekLine != null)
      trekLine.setStrokePaint( c);
    }
  
  public Color getOriginFillColor() {
    return highlightFillColor;
    }
  public void setOriginFillColor(Color c) {
    highlightFillColor = c;
    if (selectionFigure != null)
      selectionFigure.setFillPaint(c);
    }
    
  public PointsFigure getPointsFigure() {return trekLine;}
  
  public Figure getSelectionFigure() {return selectionFigure;}
  
  public String getKey() {return (String)getUserObject();}
  }
