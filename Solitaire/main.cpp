//
//  main.cpp
//  Solitaire
//
//  Created by Haruka Matsumura on 2017/07/14.
//  Copyright © 2017年 Haruka Matsumura. All rights reserved.
//

#include "../include/opencv2/highgui.hpp"
#include "../include/opencv2/imgproc.hpp"
#include "../include/opencv2/imgcodecs.hpp"
#include "../include/opencv2/core.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

class CARD{
public:
    int num;
    char type;
    cv::Mat img;
    bool visible;
    int row;
    int col;
};

struct mouseParam {
    int x;
    int y;
    char state;
};

const string imagePath = "../cards/";
const char cardType[4] = {'s', 'c', 'd', 'h'};
const int cardWidth = 409;
const int cardHeight = 600;
int reHeight = 150;
int reWidth = cardWidth / (float)cardHeight * reHeight;
const int displayWidth = 1024;
const int displayHeight = 768;
//const int dispkayHeight = 960;

vector<vector<cv::Point2f>> arrayPoint(7);
vector<cv::Point2f> foundationPoint(4);
vector<cv::Point2f> deckPoint(1);
bool clearFlag;
cv::Mat backImg, backgroundImg, moveImg;

void loadImage(vector<CARD> &cards)
{
    int i, j;
    cv::Mat frontImg;
    
    for(j = 0; j < 4; ++j)
    {
        for (i = 0; i < 13; ++i)
        {
            frontImg = cv::imread(imagePath + cardType[j] + to_string(i+1) + ".png", cv::IMREAD_UNCHANGED);
            //cout << reHeight << reWidth << endl;
            cv::resize(frontImg, frontImg, cv::Size(reWidth, reHeight));
            
            cards[j * 13 + i].num = i + 1;
            cards[j * 13 + i].type = cardType[j];
            cards[j * 13 + i].img = frontImg;
            cards[j * 13 + i].row = -1;
            cards[j * 13 + i].col = -1;
            cards[j * 13 + i].visible = true;
        }
    }
    
    backImg = cv::imread(imagePath + "back.png", cv::IMREAD_UNCHANGED);
    cv::resize(backImg, backImg, cv::Size(reWidth, reHeight));
    backgroundImg = cv::imread(imagePath + "background.png");
    moveImg = cv::imread(imagePath + "move.png");
    //background = cv::Mat(cv::Size(displayWidth, displayHeight), CV_8UC3, cv::Scalar(75, 79, 47));
}

void overlayImage(cv::Mat &baseImg, cv::Mat &transImg, vector<cv::Point2f> &tgtPt)
{
    cv::Mat img_rgb, img_aaa, img_1ma;
    vector<cv::Mat>planes_rgba, planes_rgb, planes_aaa, planes_1ma;
    int maxVal = pow(2, 8*baseImg.elemSize1())-1;
    
    tgtPt.resize(4);
    float ww  = transImg.cols;
    float hh  = transImg.rows;
    
    tgtPt[1] = cv::Point2f(tgtPt[0].x+ww, tgtPt[0].y);
    tgtPt[2] = cv::Point2f(tgtPt[0].x+ww, tgtPt[0].y+hh);
    tgtPt[3] = cv::Point2f(tgtPt[0].x   , tgtPt[0].y+hh);
    
    //変形行列を作成
    vector<cv::Point2f>srcPt;
    srcPt.push_back( cv::Point2f(0, 0) );
    srcPt.push_back( cv::Point2f(transImg.cols-1, 0) );
    srcPt.push_back( cv::Point2f(transImg.cols-1, transImg.rows-1) );
    srcPt.push_back( cv::Point2f(0, transImg.rows-1) );
    cv::Mat mat = cv::getPerspectiveTransform(srcPt, tgtPt);
    
    //出力画像と同じ幅・高さのアルファ付き画像を作成
    cv::Mat alpha0(baseImg.rows, baseImg.cols, transImg.type() );
    alpha0 = cv::Scalar::all(0);
    cv::warpPerspective(transImg, alpha0, mat, alpha0.size(), cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);
    
    //チャンネルに分解
    cv::split(alpha0, planes_rgba);
    
    //RGBA画像をRGBに変換
    planes_rgb.push_back(planes_rgba[0]);
    planes_rgb.push_back(planes_rgba[1]);
    planes_rgb.push_back(planes_rgba[2]);
    merge(planes_rgb, img_rgb);
    
    //RGBA画像からアルファチャンネル抽出
    planes_aaa.push_back(planes_rgba[3]);
    planes_aaa.push_back(planes_rgba[3]);
    planes_aaa.push_back(planes_rgba[3]);
    merge(planes_aaa, img_aaa);
    
    //背景用アルファチャンネル
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    planes_1ma.push_back(maxVal-planes_rgba[3]);
    merge(planes_1ma, img_1ma);
    
    baseImg = img_rgb.mul(img_aaa, 1.0/(double)maxVal) + baseImg.mul(img_1ma, 1.0/(double)maxVal);
}

void clearCards(vector<CARD> &cards, vector<vector<CARD>> &arrayCards, vector<vector<CARD>> &foundationCards)
{
    cards.resize(52);
    
    for (auto& c: arrayCards)
    {
        c.clear();
    }
    for (auto& c: foundationCards)
    {
        c.clear();
    }
}

void shuffleCards(vector<CARD> &cards, vector<vector<CARD>> &arrayCards)
{
    int i, j;
    random_device rd;
    mt19937 g(rd());
    shuffle(cards.begin(), cards.end(), g);
    int count = 51;
    
    for(j = 0; j < 7; ++j)
    {
        for (i = 0; i < j + 1; ++i)
        {
            if(i != j)
            {
                cards[count].visible = false;
            }else
            {
                cards[count].visible = true;
            }
            arrayCards[j].push_back(cards[count]);
            cards.pop_back();
            count--;
        }
    }
}


void drawArray(cv::Mat &displayImg, vector<vector<CARD>> &arrayCards)
{
    size_t col_index, row_index;
    vector<cv::Point2f> pos(1);
    for (auto& col: arrayCards)
    {
        col_index = &col - &arrayCards[0];
        for (auto& card: col)
        {
            row_index = &card - &col[0];
            pos[0] = arrayPoint[col_index][row_index];
            
            switch (card.visible){
                case true:
                    overlayImage(displayImg, card.img, pos);
                    break;
                case false:
                    overlayImage(displayImg, backImg, pos);
                    break;
            }
        }
    }
}

void drawFoundation(cv::Mat &displayImg, vector<vector<CARD>> &foundationCards)
{
    size_t i;
    vector<cv::Point2f> pos(1);
    for (auto& col: foundationCards)
    {
        if(col.size() != 0)
        {
            i = &col - &foundationCards[0];
            pos[0] = foundationPoint[(int)i];
            overlayImage(displayImg, col.back().img, pos);
        }
    }
}

void drawWindow(cv::Mat &background, cv::Mat &img, vector<vector<CARD>> &arrayCards, vector<vector<CARD>> &foundationCards, vector<CARD> &cards, int deckNum)
{
    background = img.clone();
    drawArray(background, arrayCards);
    drawFoundation(background, foundationCards);
    
    if(cards.size() > 0){
        overlayImage(background, cards[deckNum].img, deckPoint);
        cv::putText(background, to_string(cards.size()), cv::Point(deckPoint[0].x + reWidth + 20, deckPoint[0].y + reHeight), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 2.0, cv::Scalar(255, 255, 255), 3);
        cv::putText(background, to_string(deckNum), cv::Point(deckPoint[0].x + reWidth + 20, deckPoint[0].y + 50), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 2.0, cv::Scalar(255, 255, 255), 3);
    }
}

bool checkClickFoundationPoint(cv::Point p, int &col, int &row)
{
    cv::Rect rect;
    size_t i;
    for (auto& point: foundationPoint)
    {
        rect = cv::Rect(point.x, point.y, reWidth, reHeight);
        if(rect.contains(p))
        {
            i = &point - &foundationPoint[0];
            col = (int)i;
            row = -1;
            cout << "foundation" << i << endl;
            return true;
        }
    }
    
    return false;
}

bool checkClickArrayPoint(cv::Point p, int &col, int &row, vector<vector<CARD>> &arrayCards)
{
    cv::Rect rect;
    size_t i, j;
    for (auto& list: arrayPoint)
    {
        j = &list - &arrayPoint[0];
        for (auto& point: list)
        {
            i = &point - &list[0];
            if(i < arrayCards[j].size() - 1){
                rect = cv::Rect(point.x, point.y, reWidth, 30);
            }else if(i == arrayCards[j].size() - 1)
            {
                rect = cv::Rect(point.x, point.y, reWidth, reHeight);
            }else
            {
                break;
            }
            
            if(rect.contains(p))
            {
                col = (int)j;
                row = (int)i;
                cout << "array" << j << i << endl;
                return true;
            }
        }
    }
    return false;
}

void moveCards(vector<CARD> &fromCards, int num, vector<CARD> &toCards)// , vector<vector<CARD>> &toCards, int col)
{
    toCards.push_back(fromCards[num]);
    fromCards[num] = fromCards[fromCards.size() - 1];
    fromCards.pop_back();
}

void move2Foundation(vector<CARD> &cards, int num, vector<vector<CARD>> &foundationCards, int col)
{
    if(foundationCards[col].size() == 0)
    {
        if(cards[num].num != 1)
        {
            return;
        }else
        {
            moveCards(cards, num, foundationCards[col]);
        }
    }else if(foundationCards[col].back().num == cards[num].num - 1 && foundationCards[col].back().type == cards[num].type)
    {
        moveCards(cards, num, foundationCards[col]);
    }
}

void move2Array(vector<CARD> &cards, int num, vector<vector<CARD>> &arrayCards, int col)
{
    if(arrayCards[col].size() == 0)
    {
        if(cards[num].num != 13)
        {
            return;
        }else
        {
            moveCards(cards, num, arrayCards[col]);
        }
    }else if(arrayCards[col].back().num == cards[num].num + 1)
    {
        if((arrayCards[col].back().type == 'c' || arrayCards[col].back().type == 's') && (cards[num].type == 'h' || cards[num].type == 'd'))
        {
            moveCards(cards, num, arrayCards[col]);
        }else if((arrayCards[col].back().type == 'd' || arrayCards[col].back().type == 'h') && (cards[num].type == 'c' || cards[num].type == 's'))
        {
            moveCards(cards, num, arrayCards[col]);
        }
    }
}

bool moveArray2Array(vector<CARD> &src_cards, int &row, vector<CARD> &dst_cards)
{
    if(dst_cards.size() == 0)
    {
        if(src_cards[row].num == 13)
        {
            //moveCards(src_cards, row, dst_cards);
            dst_cards.push_back(src_cards[row]);
            row++;
            return true;
        }
    }else if(dst_cards.back().num == src_cards[row].num + 1)
    {
        if((dst_cards.back().type == 'c' || dst_cards.back().type == 's') && (src_cards[row].type == 'h' || src_cards[row].type == 'd'))
        {
            //moveCards(src_cards, row, dst_cards);
            dst_cards.push_back(src_cards[row]);
            row++;
            return true;
        }else if((dst_cards.back().type == 'h' || dst_cards.back().type == 'd') && (src_cards[row].type == 'c' || src_cards[row].type == 's'))
        {
            //moveCards(src_cards, row, dst_cards);
            dst_cards.push_back(src_cards[row]);
            row++;
            return true;
        }
    }
    return false;
}

void checkMoveCards(vector<CARD> &cards, vector<vector<CARD>> &arrayCards, vector<vector<CARD>> &foundationCards, mouseParam &mouse, int deckNum)
{
    int src_col, src_row;
    int dst_col, dst_row;
    char src_type = 'n';
    cv::Point p;
    cv::Rect rect;
    
    cv::Mat background;
    drawWindow(background, moveImg, arrayCards, foundationCards, cards, deckNum);
    cv::imshow("Solitaire", background);
    cv::waitKey(0);
    
    
    p = cv::Point(mouse.x, mouse.y);
    rect = cv::Rect(deckPoint[0].x, deckPoint[0].y, reWidth, reHeight);
    
    if(rect.contains(p) && cards.size() != 0)
    {
        cout << "deck" << endl;
        src_type = 'd';
        src_col = -1;
        src_row = -1;
    }else
    {
        if(checkClickFoundationPoint(p, src_col, src_row))
        {
            if(foundationCards[src_col].size() != 0)
            {
                src_type = 'f';
                src_row = (int)foundationCards[src_col].size();
            }
        }
        else if(checkClickArrayPoint(p, src_col, src_row, arrayCards))
        {
            src_type = 'a';
        }
    }
    
    if(src_type == 'n')
    {
        return;
    }else
    {
        cv::waitKey(0);
        p = cv::Point(mouse.x, mouse.y);

        if(checkClickFoundationPoint(p, dst_col, dst_row) && src_type != 'f')
        {
            if(src_type == 'd')
            {
                move2Foundation(cards, deckNum, foundationCards, dst_col);
            }else if(src_type == 'a' && src_row == arrayCards[src_col].size() - 1)
            {
                move2Foundation(arrayCards[src_col], src_row, foundationCards, dst_col);
                arrayCards[src_col].back().visible = true;
            }
        }
        else if(checkClickArrayPoint(p, dst_col, dst_row, arrayCards))
        {
            if(src_type == 'd')
            {
                move2Array(cards, deckNum, arrayCards, dst_col);
                //moveArray2Array(cards, deckNum, arrayCards[dst_col]);
            }else if(src_type == 'f')
            {
                move2Array(foundationCards[src_col], src_row - 1, arrayCards, dst_col);
                //moveArray2Array(foundationCards[src_col], src_row - 1, arrayCards[dst_col]);
            }else if(src_type == 'a')
            {
                int count = 0;
                int i;
                
                while(1){
                    if(!moveArray2Array(arrayCards[src_col], src_row, arrayCards[dst_col]))
                    {
                        break;
                    }
                    count++;
                    cout << arrayCards[src_col].size() << endl;
                    if(arrayCards[src_col].size() == src_row)
                    {
                        for(i = 0; i < count; ++i)
                        {
                            arrayCards[src_col].pop_back();
                        }
                        if(arrayCards[src_col].size() != 0)
                        {
                            arrayCards[src_col].back().visible = true;
                        }
                        break;
                    }
                }
            }
        }
    }
}

bool checkClear(vector<vector<CARD>> &foundationCards, vector<vector<CARD>> &arrayCards, vector<CARD> &cards)
{
    if(cards.size() != 0)
    {
        return false;
    }
    for (auto& col: foundationCards)
    {
        if(col.size() != 13)
        {
            for (auto& col: arrayCards)
            {
                for (auto& card: col)
                {
                    if (card.visible == false)
                    {
                        cout << card.type << card.num << endl;
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

char findEventType(int x, int y)
{
    int i;
    vector<cv::Rect> button(3);
    for(i = 2; i >= 0; --i)
    {
        button[i] = cv::Rect(displayWidth - (i + 1) * (reWidth + 80) , 10 + 4 * (reHeight + 20), reWidth + 60, 40);
    }
    
    char c = 'n';
    cv::Point p = cv::Point(x, y);
    
    
    if(button[0].contains(p) && !clearFlag)
    {
        c = 's';
    }else if(button[1].contains(p) && !clearFlag)
    {
        c = 'm';
    }else if(button[2].contains(p))
    {
        c = 't';
    }
    
    return c;
}

void CallBackFunc(int event, int x, int y, int flags, void* param)
{
    mouseParam *mouse = (mouseParam*)param;
    
    if (event == cv::EVENT_LBUTTONUP)
    {
        mouse->x = x;
        mouse->y = y;
        mouse->state = findEventType(x, y);
    }else if (event == cv::EVENT_RBUTTONUP)
    {
        mouse->state = 'q';
    }else
    {
        return;
    }
    cout << mouse->state << endl;
}

int main(int argc, const char * argv[])
{
    int i, j;
    
    vector<CARD> cards(52);
    vector<vector<CARD>> arrayCards(7);
    vector<vector<CARD>> foundationCards(4);
    cv::Mat background;
    
    loadImage(cards);
    background = backgroundImg.clone();
    shuffleCards(cards, arrayCards);
    clearFlag = false;
    
    for(i = 0; i < 7; ++i)
    {
        for(j = 0; j < 19; ++j)
        {
            arrayPoint[i].push_back(cv::Point2f(24 + i * (reWidth + 20), 10 + j * 30));
            if(j == 0)
            {
                //cv::rectangle(background, arrayPoint[i][j], cv::Point(arrayPoint[i][j].x + reWidth, arrayPoint[i][j].y + reHeight), cv::Scalar(128,128,128), 2);
            }
        }
    }
    
    for(i = 0; i < 4; ++i)
    {
        foundationPoint[i] = cv::Point2f(24 + 7 * (reWidth + 20), 10 + i * (reHeight + 20));
        //cv::rectangle(background, foundationPoint[i], cv::Point(foundationPoint[i].x + reWidth, foundationPoint[i].y + reHeight), cv::Scalar(128,128,128), 2);
    }
    deckPoint[0] = cv::Point2f(24 + 0 * (reWidth + 20), displayHeight - reHeight - 50);
    //cv::rectangle(background, deckPoint[0], cv::Point(deckPoint[0].x + reWidth, deckPoint[0].y + reHeight), cv::Scalar(128,128,128), 2);
    
    /*
    vector<cv::Rect> button(3);
    for(i = 2; i >= 0; --i)
    {
        button[i] = cv::Rect(displayWidth - (i + 1) * (reWidth + 80) , 10 + 4 * (reHeight + 20), reWidth + 60, 40);
        cv::rectangle(background, button[i], cv::Scalar(255, 255, 255), CV_FILLED);
    }
    */
    
    char state;
    bool initialize = true;
    int deckNum = 0;
    mouseParam mouseEvent;
    cv::namedWindow("Solitaire");
    cv::setMouseCallback("Solitaire", CallBackFunc, &mouseEvent);
    
    while(1)
    {
        cv::imshow("Solitaire", background);
        //cv::imwrite(imagePath + "background.png", background);
        cv::waitKey(0);
        if(!initialize)
        {
            state = mouseEvent.state;
            
            if(state == 'q')
            {
                break;
            }else if(state == 's')
            {
                clearCards(cards, arrayCards, foundationCards);
                loadImage(cards);
                //background = backgroundImg.clone();
                shuffleCards(cards, arrayCards);
                clearFlag = false;
            }else if(state == 't')
            {
                deckNum++;
                if(deckNum == cards.size())
                {
                    deckNum = 0;
                }
            }else if(state == 'm')
            {
                checkMoveCards(cards, arrayCards, foundationCards, mouseEvent, deckNum);
                if(deckNum == cards.size())
                {
                    deckNum = 0;
                }
            }
        }
        
        drawWindow(background, backgroundImg, arrayCards, foundationCards, cards, deckNum);
        
        if(checkClear(foundationCards, arrayCards, cards))
        {
            clearFlag = true;
            cv::putText(background, "CLEAR", cv::Point(displayWidth - 3 * (reWidth + 80), 10 + 4 * (reHeight + 20) - 20), cv::FONT_HERSHEY_COMPLEX | cv::FONT_ITALIC, 2.0, cv::Scalar(0, 0, 255), 3);
        }

        initialize = false;
    }
    return 0;
}
