/*
 * @(#)Trek.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek;

/**
 * Trek
 *
 * @version $Revision: 1.1 $
 */
public class Trek {

  protected String key;
  protected double origin[] = new double[2];
  protected double originIndependent = 0.0;
	protected double independentRange[] = new double[2];
  protected double points[][];

	public Trek(String key, double[] orig, double originIndep, double[] indepRange) {
    this.key = key;
    this.origin[0] = orig[0];
    this.origin[1] = orig[1];
    this.originIndependent = originIndep;
		this.independentRange[0] = indepRange[0];
		this.independentRange[1] = indepRange[1];
	  }

  public String getKey() {return key;}

  public double[] getOrigin() {return origin;}
  public void setOrigin(double[] neworigin) {
    origin[0] = neworigin[0];
    origin[1] = neworigin[1];
    }

	public double getOriginIndependent() {return originIndependent;}
	public void setOriginIndependent(double newVal) {
		originIndependent = newVal;
		}
		
	public double[] getIndependentRange() {return independentRange;}
	public void setIndependentRange(double[] newRange) {
		independentRange[0] = newRange[0];
		independentRange[1] = newRange[1];
		}
		
  public double[][] getPoints() {return points;}
  public void setPoints(double newPoints[][]) {
     points = newPoints;
     }
  
}