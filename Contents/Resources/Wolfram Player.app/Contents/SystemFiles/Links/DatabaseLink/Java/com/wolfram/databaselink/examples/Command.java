package com.wolfram.databaselink.examples;

import java.sql.Timestamp;

import com.odellengineeringltd.glazedlists.*;

public class Command implements Comparable, Filterable
{
  private int id;
  private String fullForm;
  private byte[] image;
  private String expr;
  private Timestamp dateTime;

  public Command(int id, String expr, String fullForm, byte[] image, Timestamp dateTime)
  {
    this.id = id;
    this.expr = expr;
    this.fullForm = fullForm;
    this.image = image;
    this.dateTime = dateTime;
  }

  public int getId()
  {
    return id;
  }

  public String getExpr()
  {
    return expr;
  }

  public String getFullForm()
  {
    return fullForm;
  }

  public byte[] getImage()
  {
    return image;
  }

  public Timestamp getDateTime()
  {
    return dateTime;
  }

  public int compareTo(Object other)
  {
    Command otherCommand = (Command)other;
    return (new Integer(getId()).compareTo(new Integer(otherCommand.getId())));
  }

  public String[] getFilterStrings()
  {
    return new String[] {fullForm};
  }

}

