/*
 * @(#)AxesFigure.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */

package com.wolfram.guikit.diva;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import javax.swing.SwingConstants;

import diva.canvas.CanvasLayer;
import diva.canvas.CanvasPane;
import diva.canvas.CompositeFigure;
import diva.canvas.JCanvas;
import diva.canvas.toolbox.LabelFigure;
import diva.canvas.toolbox.PathFigure;
import diva.util.java2d.Polyline2D;
import diva.util.java2d.ShapeUtilities;

/**
 * AxesFigure is a diva Figure subclass that represents an axes
 * 
 * @version $Revision: 1.3 $
 */
public class AxesFigure extends CompositeFigure {

  public static NumberFormat defaultDecimalFormat = NumberFormat.getInstance();
  public static NumberFormat defaultScientificDecimalFormat = new DecimalFormat("0.##E0");
  
   // TODO these need to change based on xscale and yscale and the new origin
   private double xmin,xmax,ymin,ymax; // Range of x and y values on the Rect (not counting the gap).
   private double xorigin = xmin;
   private double yorigin = ymax;
   private double xscale = 1.0;
   private double yscale = 1.0;
   
   private double gap = 5.0; //Extra pixels around the edges, outside the specifed range of x,y values.
   //Note: xmin,xmax,ymin,ymax are the limits on a rectangle that
   //is inset from the drawing rect by gap pixels on each edge.
   private int left, top, width = -1, height = -1;  // Not setable; these are valid only during drawing and are meant to be used
                                                    // by the Drawables in this Coorfdinate Rect.
                 
   private boolean needsFiguresUpdate = false;

   /**
    * A constant that can be used in the setYAxisPosition() method to indicate the placement of the y-axis.
    * The axis is placed at the top of the CoordinateRect.
    */
   public static final int TOP = 0;

   /**
    * A constant that can be used in the setYAxisPosition() method to indicate the placement of the y-axs.
    * The axis is placed at the bottom of the CoordinateRect.
    */
   public static final int BOTTOM = 1;

   /**
    * A constant that can be used in the setXAxisPosition() method to indicate the placement of the x-axis.
    * The axis is placed at the left edge of the CoordinateRect.
    */
   public static final int LEFT = 2;

   /**
    * A constant that can be used in the setXAxisPosition() method to indicate the placement of the x-axis.
    * The axis is placed at the right edge of the CoordinateRect.
    */
   public static final int RIGHT = 3;

   /**
    * A constant that can be used in the setXAxisPosition() and setYAxisPosition() methods to indicate the placement of the axes.
    * The axis is placed in the center of the CoordinateRect.
    */
   public static final int CENTER = 4;

   /**
    * A constant that can be used in the setXAxisPosition() and setYAxisPosition() methods to indicate the placement of the axes.
    * The axis is placed at its true x- or y-position, if that lies within the range of values shown on the CoordinateRect.
    * Otherwise, it is placed along an edge of the CoordinateRect.  This is the default value for axis placement.
    */
   public static final int SMART = 5;
       
   private int xAxisPosition = SMART;
   private int yAxisPosition = SMART;
   
   private Color axesColor = Color.BLACK; // new Color(0,0,180)
   // Used if real axis is outside the draw rect
   private Color lightAxesColor = Color.GRAY; // new Color(180,180,255)
   
   private Color labelColor = Color.BLACK;
   
   private String xLabel = null;
   private String yLabel = null;
      
   private transient Font font;
   
   private CompositeFigure xAxisFigure;
   private CompositeFigure yAxisFigure;
   private LabelFigure xAxisLabel;
   private LabelFigure yAxisLabel;
   
  static {
    defaultDecimalFormat.setMaximumFractionDigits(6);
    defaultDecimalFormat.setGroupingUsed(false);
    defaultScientificDecimalFormat.setMaximumFractionDigits(2);
    defaultScientificDecimalFormat.setGroupingUsed(false);
    }
    
   /*
    * Creates axes with no names on the axes.
    */
   public AxesFigure() {
      this(null,null);
      }
   
   /**
    * Creates axes with given names on the axes.
    *
    * @param xlabel   Label for x axis.  If the value is null, no label is drawn.
    * @param ylabel   Label for y axis.  If the value is null, no label is drawn.
    */
   public AxesFigure(String xLabel, String yLabel) {
      super();
      this.xLabel = xLabel;
      this.yLabel = yLabel;
      needsFiguresUpdate = true;
      }
   
    public Rectangle2D getBounds () {
      CanvasLayer layer = getLayer();
      if (layer != null) {
        CanvasPane pane = layer.getCanvasPane();
        if (pane != null) {
        	return getVisibleRect(pane);
        	//AffineTransform tr = pane.getTransformContext().getTransform();
          //Point2D p = pane.getSize();
          //return new Rectangle2D.Double(tr.getTranslateX(),tr.getTranslateY(),p.getX(),p.getY());
          }
        }
      return null;
      }
    
  public static String formatNumber(double d) {
    if (Math.abs(d) < 100000 && Math.abs(d) > 0.00001) return defaultDecimalFormat.format(d);
    else return defaultScientificDecimalFormat.format(d);
    }
  
	/** Return the size of the visible part of a canvas, in canvas
	 *  coordinates.
	 */
	public Rectangle2D getVisibleRect(CanvasPane pane) {
		if (pane == null) return null;
		AffineTransform current = pane.getTransformContext().getTransform();
		AffineTransform inverse;
		try {
			inverse = current.createInverse();
			}
		catch(NoninvertibleTransformException e) {
			throw new RuntimeException(e.toString());
			}
		JCanvas c = pane.getCanvas();
		if (c == null) return null;
		Dimension size = c.getSize();
		Rectangle2D visibleRect = new Rectangle2D.Double(0, 0, size.getWidth(), size.getHeight());
		return ShapeUtilities.transformBounds(visibleRect, inverse);
		}
	
   public void canvasReshaped(int x, int y, int w, int h) {
     left = x;
     top = y;
     width = w;
     height = h;
     
     xmax = xorigin + width/xscale;
     ymin = yorigin - height/yscale;
     
     needsFiguresUpdate = true;
     repaint();
     }
    
   public void setTransform(double xorigin, double yorigin, double xscale, double yscale) {
    this.xorigin = xorigin;
    this.yorigin = yorigin;
    this.xscale = xscale;
    this.yscale = yscale;
     
    xmin = xorigin;
    ymax = yorigin;
    xmax = xorigin + width/xscale;
    ymin = yorigin - height/yscale;
    
    needsFiguresUpdate = true;
    repaint();
    }
     
   /**
    * Get the color that is used for drawing the axes, when they are drawn in their true position.
    */
   public Color getAxesColor() {return axesColor; }
   /**
    * Set the color that is used for drawing the axes, when they are drawn in their true position.
    * The default is blue.
    */
   public void setAxesColor(Color c) { 
      if (c != null && !c.equals(axesColor)) {
         axesColor = c; 
         needsFiguresUpdate = true;
         repaint();
      }
   }
   
   /**
    * Get the color that is used for drawing an axis, when it is drawn along an edge of the CoordinateRect
    * instead of in its proper x- or y-position.
    */
   public Color getLightAxesColor() {return lightAxesColor; }
   /**
    * Get the color that is used for drawing an axis, when it is drawn along an edge of the CoordinateRect
    * instead of in its proper x- or y-position.  The default is a light blue.
    */
   public void setLightAxesColor(Color c) {
      if (c != null && !c.equals(lightAxesColor)) {
         lightAxesColor = c; 
         needsFiguresUpdate = true;
         repaint();
      }
   }

   /**
    * Get the color that is used for drawing the labels on the x- and y-axes.
    */
  public Color getLabelColor() {return labelColor; }
   /**
    * Set the color that is used for drawing the labels (usually the names of the variables) on the x- and y-axes.
    * The default is black.
    */
  public void setLabelColor(Color c) { 
    if (c != null && !c.equals(labelColor)) {
      labelColor = c; 
      if (xAxisLabel != null)
        xAxisLabel.setFillPaint(c);
      if (yAxisLabel != null)
        yAxisLabel.setFillPaint(c);
      }
   }
   
   /**
    * Get the positioning constant that tells where the x-axis is drawn.  This can be LEFT, RIGHT, CENTER, or SMART.
    */
   public int getXAxisPosition() {return xAxisPosition; }
   /**
    * Set the positioning constant that tells where the x-axis is drawn.  This can be LEFT, RIGHT, CENTER, or SMART.
    * The default is SMART.
    */
   public void setXAxisPosition(int pos) { 
       if ((pos == TOP || pos == BOTTOM || pos == CENTER || pos == SMART) && pos != xAxisPosition) {
          xAxisPosition = pos;
          needsFiguresUpdate = true;
          repaint();
       }
   }    
   
   /**
    * Get the positioning constant that tells where the y-axis is drawn.  This can be TOP, BOTTOM, CENTER, or SMART.
    */
   public int getYAxisPosition() {return yAxisPosition; }
   /**
    * Set the positioning constant that tells where the y-axis is drawn.  This can be TOP, BOTTOM, CENTER, or SMART.
    * The default is SMART.
    */
   public void setYAxisPosition(int pos) { 
       if ((pos == LEFT || pos == RIGHT || pos == CENTER || pos == SMART) && pos != yAxisPosition) {
          yAxisPosition = pos;
          needsFiguresUpdate = true;
          repaint();
       }
   }

   /**
    * Get the label that appears on the x-axis.  If the value is null, no label appears.
    */
   public String getXLabel() {return xLabel; }
   /**
    * Set the label that appears on the x-axis.  If the value is null, no label appears.  This is the default.
    */
  public void setXLabel(String s) { 
    xLabel = s; 
    if (xLabel != null) {
      needsFiguresUpdate = true;
      repaint();
      }
    else {
      if (xAxisLabel != null) {
        remove(xAxisLabel);
        xAxisLabel = null;
        }
      }
   }
   
   /**
    * Get the label that appears on the y-axis.  If the value is null, no label appears.
    */
  public String getYLabel() {return yLabel;}
   /**
    * Set the label that appears on the y-axis.  If the value is null, no label appears.  This is the default.
    */
  public void setYLabel(String s) { 
    yLabel = s;
    if (yLabel != null) {
      needsFiguresUpdate = true;
      repaint();
      }
    else {
      if (yAxisLabel != null) {
        remove(yAxisLabel);
        yAxisLabel = null;
        }
      }
    }
   
   
   /**
    * Draw the axes. This is not meant to be called directly.
    *
    */
  public void paint(Graphics2D g) {
    if (needsFiguresUpdate || !g.getFont().equals(font)) {  
      font = g.getFont();
      FontMetrics fm = g.getFontMetrics(font);
      updateFigures(fm, xmin, xmax, ymin, ymax, left, top, width, height, gap);
      needsFiguresUpdate = false;
      }
    super.paint(g);
    }
   
  protected void updateFigures(FontMetrics fm, double xmin, double xmax, double ymin, double ymax,
      int left, int top, int width, int height, double gap) {
    if (fm == null) return;
    
    double axisPositionX = 0;
    double axisPositionY = 0;
    int ascent = fm.getAscent();
    int descent = fm.getDescent();
    int digitWidth = fm.charWidth('0');
    Color useColor = Color.WHITE;

    switch (xAxisPosition) {
       case TOP: 
          axisPositionX = top + gap; 
          break;
       case BOTTOM: 
          axisPositionX = top + height - gap - 1; 
          break;
       case CENTER: 
          axisPositionX = top + height/2.0; 
          break;
       case SMART:
          // TODO if we ever support an alternate axes origin from (0,0) these change
          //   and elsewhere
          if (ymax < 0)
             axisPositionX = top + gap;
          else if (ymin > 0)
             axisPositionX = top + height - gap - 1;
          else
             axisPositionX = top + gap + (height-2.0*gap - 1) * ymax / (ymax-ymin);
          break;
      }
    switch (yAxisPosition) {
       case LEFT: 
          axisPositionY = left + gap; 
          break;
       case BOTTOM: 
          axisPositionY = left + width - gap - 1; 
          break;
       case CENTER: 
          axisPositionY = left + width/2.0; 
          break;
       case SMART:
          // TODO if we ever support an alternate axes origin from (0,0) these change
          //   and elsewhere
          if (xmax < 0)
             axisPositionY = left + width - gap - 1;
          else if (xmin > 0)
             axisPositionY = left + gap;
          else
             axisPositionY = left + gap - (width-2.0*gap - 1) * xmin / (xmax-xmin);
          break;
      }

    double start = tweakStart( 
      ((xmax-xmin)*(axisPositionY - (left + gap)))/(width - 2*gap)  + xmin, 
      0.05*(xmax-xmin) );
    int labelCt = (int)((width - 2*gap) / (10*digitWidth));
    if (labelCt <= 2)
       labelCt = 3;
    else if (labelCt > 20)
       labelCt = 20;
    double interval = tweak( (xmax-xmin)/labelCt );
    for (double mul = 1.5; mul < 4; mul += 0.5) {
       if (fm.stringWidth(formatNumber(interval+start)) + 
             digitWidth > (interval/(xmax-xmin))*(width-2*gap))  // overlapping labels
           interval = tweak( mul*(xmax - xmin) / labelCt );
       else
          break;
      }
    double[] label = new double[50];
    labelCt = 0;
    double x = start + interval;
    double limit = left + width;
    if (xLabel != null && 
          left + width - gap - fm.stringWidth(xLabel) > axisPositionY) // avoid overlap with xLabel
       limit -= fm.stringWidth(xLabel) + gap + digitWidth;
    while (labelCt < 50 && x <= xmax) {
       if (left + gap + (width-2.0*gap)*(x-xmin)/(xmax-xmin) + 
            fm.stringWidth(formatNumber(x))/2.0 > limit)
          break;
       label[labelCt] = x;
       labelCt++;
       x += interval;
      }
    x = start - interval;
    limit = left;
    if (xLabel != null && 
        left + width - gap - fm.stringWidth(xLabel) <= axisPositionY)  // avoid overlap with xLabel
       limit += fm.stringWidth(xLabel) + digitWidth;
    while (labelCt < 50 && x >= xmin) {
       if (left + gap + (width-2.0*gap)*(x-xmin)/(xmax-xmin) - 
           fm.stringWidth(formatNumber(x))/2.0 < limit)
          break;
       label[labelCt] = x;
       labelCt++;
       x -= interval;
       }
       
    if (xAxisPosition == SMART && (ymax < 0 || ymin > 0))
       useColor = lightAxesColor;
    else 
       useColor = axesColor;
       
    if (xAxisFigure != null)
      remove(xAxisFigure);
    xAxisFigure = new CompositeFigure(); 
    xAxisFigure.add( createLine(useColor, 
      left + gap, axisPositionX, left + width - gap - 1, axisPositionX));    
       
    double tick;
    double tickLabelPosX;
    double tickLabelPosY;
    // orig above axes labels
    //if (xAxisPixelPosition - 4 - ascent >= top)
    //   tickLabelPosY = xAxisPixelPosition - 4;
    //else
    //   tickLabelPosY = xAxisPixelPosition + 4 + ascent;
    if (axisPositionX + 3 + ascent + descent + gap >= top + height)
      tickLabelPosY = axisPositionX - 4;
    else
      tickLabelPosY = axisPositionX + 3 + ascent;
    double a = (axisPositionX - 2 < top) ? axisPositionX : axisPositionX - 2;
    double b = (axisPositionX + 2 >= top + height)? axisPositionX : axisPositionX + 2; 
    String tickLabel;
    for (int i = 0; i < labelCt; i++) {
       tick = left + gap + (width-2*gap)*(label[i]-xmin)/(xmax-xmin);
       tickLabel = formatNumber(label[i]);
       tickLabelPosX = tick - fm.stringWidth(tickLabel)/2;
       xAxisFigure.add(createLine(useColor, tick, a, tick, b));
       xAxisFigure.add(createString(tickLabel, fm.getFont(), useColor, tickLabelPosX, tickLabelPosY));
       }
    add(xAxisFigure);
    
    start = tweakStart(
      ymax - ((ymax-ymin)*(axisPositionX - (top + gap)))/(height - 2*gap), 
      0.05*(ymax-ymin) );
      
    labelCt = (int)((height - 2*gap) / (5*(ascent+descent)));
    if (labelCt <= 2)
       labelCt = 3;
    else if (labelCt > 20)
       labelCt = 20;
    interval = tweak( (ymax - ymin) / labelCt );
    labelCt = 0;
    double y = start + interval;
    limit = top + 8 + gap;
    if (yLabel != null && top + gap + ascent + descent <= axisPositionX)  // avoid overlap with yLabel
        limit = top + gap + ascent + descent;
    while (labelCt < 50 && y <= ymax) {
       if (top + gap + (height-2*gap)*(ymax-y)/(ymax-ymin) - ascent/2 < limit)
          break;
       label[labelCt] = y;
       labelCt++;
       y += interval;
      }
    y = start - interval;
    limit = top + height - gap - 8;
    if (yLabel != null && top + gap + ascent + descent > axisPositionX)  // avoid overlap with yLabel
        limit = top + height - gap - ascent - descent;
    while (labelCt < 50 && y >= ymin) {
       if (top + gap + (height-2*gap)*(ymax-y)/(ymax-ymin) + ascent/2 > limit)
          break;
       label[labelCt] = y;
       labelCt++;
       y -= interval;
       }
    
    if (yAxisPosition == SMART && (xmax < 0 || xmin > 0))
       useColor = lightAxesColor;
    else 
       useColor = axesColor;
     
    if (yAxisFigure != null)
      remove(yAxisFigure);
    yAxisFigure = new CompositeFigure();
    yAxisFigure.add(createLine(useColor,
      axisPositionY, top + gap, axisPositionY, top + height - gap - 1));
    
    String[] yTickLabels = new String[labelCt];
    int w = 0;  // max width of tick mark
    for (int i = 0; i < labelCt; i++) {
      yTickLabels[i] = formatNumber(label[i]);
      int s = fm.stringWidth(yTickLabels[i]);
      if (s > w) w = s;  
      }
    a = (axisPositionY - 2 < left) ? axisPositionY : axisPositionY - 2;
    b = (axisPositionY + 2 >= left + width)? axisPositionY : axisPositionY + 2; 
    for (int i = 0; i < labelCt; i++) {
      tickLabel = yTickLabels[i];
      tick = top + gap + (height-2*gap)*(ymax-label[i])/(ymax-ymin);
      tickLabelPosY = tick + ascent/2;
      if (axisPositionY - 4 - w < left)
        tickLabelPosX = axisPositionY + 4;
      else
        tickLabelPosX = axisPositionY - 4 - fm.stringWidth(tickLabel);
      yAxisFigure.add(createLine(useColor, a, tick, b, tick));
      yAxisFigure.add(createString(tickLabel, fm.getFont(), useColor, tickLabelPosX, tickLabelPosY));
      }
      
    add(yAxisFigure);

    if (xAxisLabel != null)
      remove(xAxisLabel);
    if (xLabel != null) {
       double xLabel_x = 0, xLabel_y = 0;
       int size = fm.stringWidth(xLabel);
       if (left + width - gap - size <= axisPositionY)
          xLabel_x = left + gap;
       else
          xLabel_x = left + width - gap - size;
       // orig below axes label
       //if (xAxisPixelPosition + 3 + ascent + descent + gap >= top + height)
       //   xLabel_y = xAxisPixelPosition - 4;
       //else
       //   xLabel_y = xAxisPixelPosition + 3 + ascent;
       if (axisPositionX - 4 - ascent >= top)
          xLabel_y = axisPositionX - 4;
       else
          xLabel_y = axisPositionX + 4 + ascent;

      xAxisLabel = createString(xLabel, fm.getFont(), labelColor, xLabel_x, xLabel_y);
      add(xAxisLabel);
      }
    
    if (yAxisLabel != null) 
      remove(yAxisLabel);
    if (yLabel != null) {
      double yLabel_x = 0, yLabel_y = 0;
      int size = fm.stringWidth(yLabel);
       if (axisPositionY + 3 + size + gap > left + width)
          yLabel_x = axisPositionY - size - 3;
       else
          yLabel_x = axisPositionY + 3;
       if (top + ascent + descent + gap > axisPositionX)
          yLabel_y = top + height - gap - descent;
       else
          yLabel_y = top + ascent + gap;
      yAxisLabel = createString(yLabel, fm.getFont(), labelColor, yLabel_x, yLabel_y);
      add(yAxisLabel);
      }
         
   }
   
   
   private PathFigure createLine(Color c, double x1, double y1, double x2, double y2) {
      Polyline2D p = new Polyline2D.Double(2);
      p.moveTo(x1, y1);
      p.lineTo(x2, y2);
  
      PathFigure pathF = new PathFigure(p);
      pathF.setLineWidth(0.5f);
      pathF.setStrokePaint( c);
      return pathF;
      }
   
   private LabelFigure createString(String text, Font f, Color c, double x1, double y1) {
      LabelFigure fig = new LabelFigure(text, f, 0.0, SwingConstants.SOUTH_WEST);
      fig.setFillPaint(c);
      fig.translateTo(new Point2D.Double(x1, y1));
      return fig;
    }
      
   /*
    * Move x to a more "rounded" value; used for labeling axes.
    */
   private double tweak(double x) { 
    int digits;
    double y;
    
    if (Math.abs(x) < 0.0005 || Math.abs(x) > 500000)
      return x;
    else if (Math.abs(x) < 0.1 || Math.abs(x) > 5000) {
      y = x;
      digits = 0;
      if (Math.abs(y) >= 1) {
        while (Math.abs(y) >= 8.75) {
          y = y / 10;
          digits = digits + 1;
          }
        }
      else {
        while (Math.abs(y) < 1) {
          y = y * 10;
          digits = digits - 1;
          }
        }
      y = Math.round(y * 4) / 4;
      if (digits > 0) {
        for (int j = 0; j < digits; j++)
          y = y * 10;
        }
      else if (digits < 0) {
        for (int j = 0; j < -digits; j++)
          y = y / 10;
        }
      return y;
      }
    else if (Math.abs(x) < 0.5)
      return Math.round(10 * x) / 10.0;
    else if (Math.abs(x) < 2.5)
      return Math.round(2 * x) / 2.0;
    else if (Math.abs(x) < 12)
      return Math.round(x);
    else if (Math.abs(x) < 120) 
      return Math.round(x / 10) * 10.0;
    else if (Math.abs(x) < 1200)
      return Math.round(x / 100) * 100.0;
    else
      return Math.round(x / 1000) * 1000.0;
    }
   
   private double tweakStart(double a, double diff) { 
    // Tries to find a "rounded value" within diff of a.
    if (Math.abs(Math.round(a) - a) < diff)
      return Math.round(a);
    for (double x = 10; x <= 100000; x *= 10) {
      double d = Math.round(a*x) / x;
      if (Math.abs(d - a) < diff)
        return d;
      }
    return a;
    }
 
  }
