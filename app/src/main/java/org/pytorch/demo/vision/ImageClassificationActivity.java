package org.pytorch.demo.vision;

import android.os.Bundle;
import android.os.SystemClock;
import android.os.Vibrator;
import android.content.Context;
import android.text.TextUtils;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.view.Gravity;
import android.widget.TextView;
import android.widget.ImageView;
import android.widget.Toast;
import android.widget.Button;
import android.widget.EditText;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;
import java.util.Arrays;
import java.util.Collections;
import java.lang.Math;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

  public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
  public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

  private static final int INPUT_TENSOR_WIDTH = 224;
  private static final int INPUT_TENSOR_HEIGHT = 224;
  private static final int TOP_K = 3;
  private static final int MOVING_AVG_PERIOD = 10;
  private static final String FORMAT_MS = "%dms";
  private static final String FORMAT_AVG_MS = "avg:%.0fms";
  private static final String FORMAT_INFO_MAX = "max:%.4f";
  private static final String FORMAT_INFO_MIN = "min:%.4f";
  private static final String FORMAT_INFO = "i:%d";
  private static final String FORMAT_INFO_CRITERIONCNT = "cricnt:%d";
  private static final String FORMAT_INFO_CRITERIONNUM = "crinum:%f";

  private static final String FORMAT_FPS = "%.1fFPS";
  public static final String SCORES_FORMAT = "%.2f";

  static class AnalysisResult {

    private final float[] depthmap;
    private final long analysisDuration;
    private final long moduleForwardDuration;

    public AnalysisResult(float[] depthmap,
                          long moduleForwardDuration, long analysisDuration) {
      this.depthmap = depthmap;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
    }
  }
  private Bitmap mBitmap;
  private ImageView mImageView;
  private boolean mAnalyzeImageErrorState;
  private ResultRowView[] mResultRowViews = new ResultRowView[TOP_K];
  private TextView mFpsText;
  private TextView mMsText;
  private TextView mMsAvgText;
  private TextView mInfoMaxText;
  private TextView mInfoMinText;
  private TextView mInfoText;
  private TextView mInfoCriterionCnt;
  private TextView mInfoCriterionNum;

  private EditText criterion_cnt_text;
  private Button criterion_cnt_btn;
  private int criterion_cnt = 2000;

  private EditText criterion_num_text;
  private Button criterion_num_btn;
  private float criterion_num = 1.0f;

  private Module mModule;
  private String mModuleAssetName;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private long mMovingAvgSum = 0;
  private Queue<Long> mMovingAvgQueue = new LinkedList<>();
  private Vibrator vibrator;
  private boolean prev_vibrate = false;

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_classification;
  }

  @Override
  protected TextureView getCameraPreviewTextureView() {
    return findViewById(R.id.image_classification_texture_view);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    mFpsText = findViewById(R.id.image_classification_fps_text);
    mMsText = findViewById(R.id.image_classification_ms_text);
    mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);
    mInfoMaxText = findViewById(R.id.info_max_text);
    mInfoMinText = findViewById(R.id.info_min_text);
    mInfoText = findViewById(R.id.info_text);
    mInfoCriterionCnt = findViewById(R.id.info_criterion_cnt);
    mInfoCriterionNum = findViewById(R.id.info_criterion_num);
  }
  protected float getMaxValue(float[] numbers){
    float maxValue = numbers[0];
    for(int i=1;i < numbers.length;i++){
      if(numbers[i] > maxValue){
        maxValue = numbers[i];
      }
    }
    return maxValue;
  }
  protected float getMinValue(float[] numbers){
    float minValue = numbers[0];
    for(int i=1;i<numbers.length;i++){
      if(numbers[i] < minValue){
        minValue = numbers[i];
      }
    }
    return minValue;
  }

  protected int[] FloatToBitmap(float[] float_array, int width, int height){
    float minValue = getMinValue(float_array);
    float maxValue = getMaxValue(float_array);

    int[] imap = new int [width*height];
    for (int i = 0; i < width*height; i++){
      imap[i] = Math.round((int) ((float_array[i] - minValue) * 255. / (maxValue - minValue)));
    }

    return imap;
  }

  @Override
  protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
    final float[] depthmap = result.depthmap;

    mImageView = findViewById(R.id.imageView);

    int width = 224;//mBitmap.getWidth();
    int height = 224;//mBitmap.getHeight();


    int[] imap = FloatToBitmap(depthmap, width, height);
    Log.i("MyActivity", "Before outbitmap");
    Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);//Bitmap.createBitmap(imap, width, height, this.getConfig());//bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
    System.out.print("imap[0:" + imap.length + "]: ");
    for (int i = 0; i < 100; i++){
      System.out.print(imap[i] + " ");
    }
    System.out.println();

    for (int i = 0; i < width; i++){
      for (int j = 0; j < height; j++) {
        int color = (255 & 0xff) << 24 | (imap[i*width + j] & 0xff) << 16 | (imap[i*width + j] & 0xff) << 8 | (imap[i*width + j] & 0xff);
        outputBitmap.setPixel(j, i, color);
      }
    }
    Log.i("MyActivity", "After outbitmap");
    mImageView.setImageBitmap(outputBitmap);

    System.out.println("width=" + outputBitmap.getWidth());
    System.out.println("height=" + outputBitmap.getHeight());
    System.out.println("pix[0][0]=" + outputBitmap.getPixel(0, 0));
    System.out.println("pix[0][1]=" + outputBitmap.getPixel(0, 1));
    System.out.println("pix[1][0]=" + outputBitmap.getPixel(1, 0));
    System.out.println("pix[1][1]=" + outputBitmap.getPixel(1, 1));
    System.out.println("pix[100][100]=" + outputBitmap.getPixel(100, 100));
//    for (int i = 0; i < 10; i++){
//      for (int j = 0; j < 10; j++) {
//        System.out.print(outputBitmap.getPixel(i, j) + " ");
//      }
//    }
    System.out.println();

    mMovingAvgSum += result.moduleForwardDuration;
    mMovingAvgQueue.add(result.moduleForwardDuration);
    if (mMovingAvgQueue.size() > MOVING_AVG_PERIOD) {
      mMovingAvgSum -= mMovingAvgQueue.remove();
    }

    // infos: fps, avgsec/image
    mMsText.setText(String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration));
    if (mMsText.getVisibility() != View.VISIBLE) {
      mMsText.setVisibility(View.VISIBLE);
    }
    mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
    if (mFpsText.getVisibility() != View.VISIBLE) {
      mFpsText.setVisibility(View.VISIBLE);
    }

    if (mMovingAvgQueue.size() == MOVING_AVG_PERIOD) {
      float avgMs = (float) mMovingAvgSum / MOVING_AVG_PERIOD;
      mMsAvgText.setText(String.format(Locale.US, FORMAT_AVG_MS, avgMs));
      if (mMsAvgText.getVisibility() != View.VISIBLE) {
        mMsAvgText.setVisibility(View.VISIBLE);
      }
    }

    // notification
    vibrator = (Vibrator)getSystemService(Context.VIBRATOR_SERVICE);
    int cnt = 0;
    for (int i = 0; i < width; i++){
      for (int j = 0; j < height; j++) {
        if (depthmap[i*width + j] < criterion_num){
          cnt += 1;
        }
      }
    }
    if (cnt >= criterion_cnt){
      if (prev_vibrate == false) {
        Toast toast = Toast.makeText(ImageClassificationActivity.this, "Vibrate", 100);
        toast.setGravity(Gravity.TOP | Gravity.RIGHT, 0, 0);
        toast.show();
      }
      vibrator.vibrate(100);
      prev_vibrate = true;
    }
    else{
      prev_vibrate = false;
    }
    mInfoMaxText.setText(String.format(Locale.US, FORMAT_INFO_MAX, getMaxValue(depthmap)));
    if (mInfoMaxText.getVisibility() != View.VISIBLE) {
      mInfoMaxText.setVisibility(View.VISIBLE);
    }
    mInfoMinText.setText(String.format(Locale.US, FORMAT_INFO_MIN, getMinValue(depthmap)));
    if (mInfoMinText.getVisibility() != View.VISIBLE) {
      mInfoMinText.setVisibility(View.VISIBLE);
    }
    mInfoText.setText(String.format(Locale.US, FORMAT_INFO, cnt));
    if (mInfoText.getVisibility() != View.VISIBLE) {
      mInfoText.setVisibility(View.VISIBLE);
    }
    mInfoCriterionCnt.setText(String.format(Locale.US, FORMAT_INFO_CRITERIONCNT, criterion_cnt));
    if (mInfoCriterionCnt.getVisibility() != View.VISIBLE) {
      mInfoCriterionCnt.setVisibility(View.VISIBLE);
    }

    criterion_cnt_text = (EditText) findViewById(R.id.criterion_cnt_text);
    criterion_cnt_btn = (Button) findViewById(R.id.criterion_cnt_btn);  //더하기 버튼(개별뷰) 연결

    criterion_cnt_btn.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {

        if(criterion_cnt_text.getText().toString().equals("")) {
          Toast.makeText(ImageClassificationActivity.this,
                  "no values in text",
                  Toast.LENGTH_LONG).show();
        } else { //둘 다 공백이 아닐때
          String s1 = criterion_cnt_text.getText().toString();//1번 값 가져오기(xml->java)
          criterion_cnt = Integer.parseInt(s1);
          criterion_cnt_text.setText(s1);
        }
      }
    });

    mInfoCriterionNum.setText(String.format(Locale.US, FORMAT_INFO_CRITERIONNUM, criterion_num));
    if (mInfoCriterionNum.getVisibility() != View.VISIBLE) {
      mInfoCriterionNum.setVisibility(View.VISIBLE);
    }

    criterion_num_text = (EditText) findViewById(R.id.criterion_num_text);
    criterion_num_btn = (Button) findViewById(R.id.criterion_num_btn);  //더하기 버튼(개별뷰) 연결

    criterion_num_btn.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {

        if(criterion_num_text.getText().toString().equals("")) {
          Toast.makeText(ImageClassificationActivity.this,
                  "no values in text",
                  Toast.LENGTH_LONG).show();
        } else { //둘 다 공백이 아닐때
          String s1 = criterion_num_text.getText().toString();//1번 값 가져오기(xml->java)
          criterion_num = Float.parseFloat(s1);
          criterion_num_text.setText(s1);
        }
      }
    });
  }

  protected String getModuleAssetName() {
    if (!TextUtils.isEmpty(mModuleAssetName)) {
      return mModuleAssetName;
    }
    final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
    mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
        ? moduleAssetNameFromIntent
        : "resnet18.pt";

    return mModuleAssetName;
  }

  @Override
  protected String getInfoViewAdditionalText() {
    return getModuleAssetName();
  }

  @Override
  @WorkerThread
  @Nullable
  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

//    try {
//      if (mModule == null) {
//        module = Module.load(assetFilePath(this, "deeplabv3_scripted.pt"));
//      }
//    } catch (IOException e) {
//      Log.e("ImageSegmentation", "Error loading model!", e);
//      finish();
//    }

    try {
      if (mModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
            Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
        Log.i("MyActivity", "Before loading model");
//        mModule = Module.load(moduleFileAbsoluteFilePath);
        mModule = Module.load(Utils.assetFilePath(this, "mobilenet-nnconv5.pth.tar.ptl"));
        Log.i("MyActivity", "After loading model");

        mInputTensorBuffer =
            Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
        mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});
      }

      final long startTime = SystemClock.elapsedRealtime();
      TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
          image.getImage(), rotationDegrees,
          INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
          TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
          TensorImageUtils.TORCHVISION_NORM_STD_RGB,
          mInputTensorBuffer, 0);

      final long moduleForwardStartTime = SystemClock.elapsedRealtime();
      final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
      final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

      final float[] depthmap = outputTensor.getDataAsFloatArray();
      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult(depthmap, moduleForwardDuration, analysisDuration);
    } catch (Exception e) {
      Log.e(Constants.TAG, "Error during image analysis", e);
      mAnalyzeImageErrorState = true;
      runOnUiThread(() -> {
        if (!isFinishing()) {
          showErrorDialog(v -> ImageClassificationActivity.this.finish());
        }
      });
      return null;
    }
  }

  @Override
  protected int getInfoViewCode() {
    return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (mModule != null) {
      mModule.destroy();
    }
  }
}
