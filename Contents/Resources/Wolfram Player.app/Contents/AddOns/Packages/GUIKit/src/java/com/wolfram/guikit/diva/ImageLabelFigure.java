/*
 * @(#)ImageLabelFigure.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.diva;

import diva.canvas.CanvasUtilities;
import diva.canvas.toolbox.LabelFigure;

import java.awt.Image;
import java.awt.Shape;
import java.awt.Graphics2D;

import java.awt.geom.Rectangle2D;
import java.awt.geom.Point2D;
import java.awt.geom.AffineTransform;

import javax.swing.SwingConstants;

/**
 * ImageLabelFigure is a subclass of LabelFigure that displays an image and not
 * text string
 */
public class ImageLabelFigure extends LabelFigure {

    /**
     * The local transform
     */
    private AffineTransform _xf = new AffineTransform();
    
    /**
     * The image of this figure.
     */
    private Image _image;
    
    /** The anchor on the label. This must be one of the
     * constants defined in SwingConstants.
     */
    private int _anchor = SwingConstants.CENTER;

    /** The "padding" around the image
     */
    private double _padding = 4.0;

    /** The order of anchors used by the autoanchor method.
     */
    private static int _anchors[] = {
        SwingConstants.SOUTH,
        SwingConstants.NORTH,
        SwingConstants.WEST,
        SwingConstants.EAST,
        SwingConstants.SOUTH_WEST,
        SwingConstants.SOUTH_EAST,
        SwingConstants.NORTH_WEST,
        SwingConstants.NORTH_EAST
    };

    /**
     * Construct an empty label figure.
     */
    public ImageLabelFigure() {
      this(null);
      }

    /**
     * Construct a label figure displaying the
     * given string, using the default font.
     */
    public ImageLabelFigure(Image i) {
      super();
      setImage(i);
      }

    /**
     * Return the figure's image.
     */
    public Image getImage() {
        return _image;
    }

    /**
     * Return the rectangular shape of the
     * image, or a small rectangle if the
     * image is null.
     */
    public Shape getShape() {
        if(_image != null) {
            int w = _image.getWidth(null);
            int h = _image.getHeight(null);
            Rectangle2D r = new Rectangle2D.Double(0, 0, w, h);
            return _xf.createTransformedShape(r);
        }
        else {
            return new Rectangle2D.Double();
        }
    }

    /**
     * Paint the figure's image.
     */
    public void paint(Graphics2D g) {
        if (!isVisible()) {
             return;
        }
        if(_image != null) {
          g.drawImage(_image, _xf, null);
          }
    }

    /** Get the bounding box of this figure.  This default
     * implementation returns the bounding box of the figure's
     * outline shape.
     */
    public Rectangle2D getBounds () {
        return getShape().getBounds2D();
    }
    
    
    /**
     * Set the figure's image.
     */
    public void setImage(Image i) {

      // repaint the string where it currently is
      repaint();

      // Remember the current anchor point
      Point2D pt = getAnchorPoint();
      
      // Modify the image
      _image = i;

      // Recalculate and translate
      Point2D badpt = getAnchorPoint();

      translate(pt.getX() - badpt.getX(), pt.getY() - badpt.getY());

      // Repaint in new location
      repaint();
      }

    /**
     * Perform an affine transform on this
     * image.
     */
    public void transform(AffineTransform t) {
        repaint();
        _xf.preConcatenate(t);
        repaint();
    }
    
    /** Choose an anchor point so as not to intersect a given
     * figure. The anchor point is cycled through until one is reached
     * such that the bounding box of the label does not intersect
     * the given shape.  If there is none,
     * the anchor is not changed. The order of preference is the
     * current anchor, the four edges, and the four corners.
     */
    public void autoAnchor (Shape s) {
        Rectangle2D.Double r = new Rectangle2D.Double();
        r.setRect(getBounds());

        // Try every anchor and if there's no overlap, use it
        Point2D location = getAnchorPoint();
        for (int i = 0; i < _anchors.length; i++) {
            Point2D pt = CanvasUtilities.getLocation(r, _anchors[i]);
            CanvasUtilities.translate(pt, _padding, _anchors[i]);
            r.x += location.getX() - pt.getX();
            r.y += location.getY() - pt.getY();
            if (!s.intersects(r)) {
                //// System.out.println("Setting anchor to " + _anchors[i]);
                setAnchor(_anchors[i]);
                break;
            }
        }
    }

    /**
     * Get the point at which this figure is "anchored." This
     * will be one of the positioning constants defined in
     * javax.swing.SwingConstants.
     */
    public int getAnchor () {
        return _anchor;
    }

    /**
     * Get the location at which the anchor is currently located.
     * This method looks at the anchor and padding attributes to
     * figure out the point.
     */
    public Point2D getAnchorPoint () {
        Rectangle2D bounds = getBounds();
        Point2D pt = CanvasUtilities.getLocation(bounds, _anchor);
        if (_anchor != SwingConstants.CENTER) {
            CanvasUtilities.translate(pt, _padding, _anchor);
        }
        return pt;
    }

    /** Return the origin, which is the anchor point.
     *  @return The anchor point.
     */
    public Point2D getOrigin () {
        return getAnchorPoint();
    }

    /**
     * Get the padding around the text.
     */
    public double getPadding () {
        return _padding;
    }

    /**
     * Set the point at which this figure is "anchored." This
     * must be one of the positioning constants defined in
     * javax.swing.SwingConstants. The default is
     * SwingConstants.CENTER. Whenever the font or string is changed,
     * the label will be moved so that the anchor remains at
     * the same position on the screen. When this method is called,
     * the figure is adjusted so that the new anchor is at the
     * same position as the old anchor was. The actual position of
     * the text relative to the anchor point is shifted by the
     * padding attribute.
     */
    public void setAnchor (int anchor) {
	  Point2D oldpt = getAnchorPoint();
	  _anchor = anchor;
	  Point2D newpt = getAnchorPoint();

	  repaint();
	  translate(
		    oldpt.getX() - newpt.getX(),
		    oldpt.getY() - newpt.getY());
	  repaint();

    }


    /**
     * Set the "padding" around the text. This is used
     * only if anchors are used -- when the label is positioned
     * relative to an anchor, it is also shifted by the padding
     * distance so that there is some space between the anchor
     * point and the text. The default padding is two, and the
     * padding must not be set to zero if automatic anchoring
     * is used.
     */
    public void setPadding (double padding) {
        _padding = padding;
        setAnchor(_anchor);
    }


    /**
     * Translate the label so that the current anchor is located
     * at the given point. Use this if you apply a transform to
     * a label in order to rotate or scale it, but don't want
     * the label to actually go anywhere.
     */
    public void translateTo (double x, double y) {
        // FIXME: this might not work in the presence of
        // scaling. If not, modify to preconcatenate instead
        repaint();
        Point2D pt = getAnchorPoint();
        translate(x-pt.getX(),y-pt.getY());
        repaint();
     }

    /**
     * Translate the label so that the current anchor is located
     * at the given point. Use this if you apply a transform to
     * a label in order to rotate or scale it, but don't want
     * the label to actually go anywhere.
     */
    public void translateTo (Point2D pt) {
        translateTo(pt.getX(), pt.getY());
     }

}