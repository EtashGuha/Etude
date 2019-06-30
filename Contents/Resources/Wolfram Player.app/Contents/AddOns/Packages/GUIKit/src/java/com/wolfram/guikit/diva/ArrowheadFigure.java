/*
 * @(#)ArrowheadFigure.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.diva;

import diva.canvas.AbstractFigure;
import diva.util.java2d.Polygon2D;

import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;

/** 
 * ArrowheadFigure is drawn on the end of a connector.
 */
public class ArrowheadFigure extends AbstractFigure {

    /** The arrowhead length, and its x and y components
     */
    private double _length = 12.0;

    /** x and y-origins
     */
    private double _originX = 0.0;
    private double _originY = 0.0;

    /** The normal to the line
     */
    private double _normal = 0.0;

    /** The shape to draw
     */
    private Polygon2D _polygon = null;

    /** A flag that says whether the shape is valid
     */
    private boolean _polygonValid = false;


    /**
     * Create a new arrowhead at (0,0).
     */
    public ArrowheadFigure() {
      this(0.0,0.0,0.0);
      }

    /**
     * Create a new arrowhead at the given point and
     * with the given normal.
     */
    public ArrowheadFigure(double x, double y, double normal) {
        _originX = x;
        _originY = y;
        _normal = normal;
        reshape();
    }

    public Shape getShape () {
      return _polygon;
      }
      
    /** Get the bounding box of the shape used to draw
     * this connector end.
     */
    public Rectangle2D getBounds () {
      return _polygon.getBounds2D();
      }

    /** Get the origin into the given point.
     */
    public void getOrigin (Point2D p) {
        p.setLocation(_originX, _originY);
    }

    /** Get the length.
     */
    public double getLength () {
        return _length;
    }

    /** Paint the arrow-head.  This method assumes that
     * the graphics context is already set up with the correct
     * paint and stroke.
     */
    public void paint (Graphics2D g) {
      if (!isVisible()) return;
      
      if (!_polygonValid) {
        reshape();
        }
      g.fill(_polygon);
      }

    /** Recalculate the shape of the decoration.
     */
    public void reshape () {
        AffineTransform at = new AffineTransform();
        at.setToRotation(_normal, _originX, _originY);

        double l1 = _length * 1.0;
        double l2 = _length * 1.3;
        double w = _length * 0.4;

        _polygon = new Polygon2D.Double() {
           public Rectangle2D getBounds2D() {
            // There is still some repainting dirty from arrowheads related to
            // the rotated coord calcs??
            Rectangle2D basicBounds = super.getBounds2D();
            return new Rectangle2D.Double(basicBounds.getX() - 5.0, 
              basicBounds.getY() - 5.0, basicBounds.getWidth() + 10.0, basicBounds.getHeight() + 10.0);
            }
            };
        _polygon.moveTo(_originX, _originY);
        _polygon.lineTo(
                _originX + l2,
                _originY + w);
        _polygon.lineTo(
                _originX + l1,
                _originY);
        _polygon.lineTo(
                _originX + l2,
                _originY - w);
        _polygon.closePath();
        _polygon.transform(at);
    }

    /** Set the normal of the decoration. The argument is the
     * angle in radians away from the origin. The arrowhead is
     * drawn so that the body of the arrowhead is in the
     * same direction as the normal -- that is, the arrowhead
     * appears to be pointed in the opposite direction to its
     * "normal."
     */
    public void setNormal (double angle) {
        _normal = angle;
        _polygonValid = false;
    }

    /** Set the origin of the decoration.
     */

    public void setOrigin(double x, double y) {
      translate(x - _originX, y - _originY);
      }
      
    /** Set the length of the arrowhead.
     */
    public void setLength(double l) {
        _length = l;
        _polygonValid = false;
    }

    public void transform (AffineTransform at) {
      _polygon.transform(at);
      }
      
    /** Translate the origin by the given amount.
     */
    public void translate (double x, double y) {
        _originX += x;
        _originY += y;

        if (_polygonValid) {
            _polygon.translate(x, y);
        }
    }
}



