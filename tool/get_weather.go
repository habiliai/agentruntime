package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/habiliai/agentruntime/entity"
	"github.com/pkg/errors"
)

type (
	GetWeatherRequest struct {
		Location string `json:"location" jsonschema:"required,description=Location to get the weather for"`
		Date     string `json:"date" jsonschema:"required,description=Date to get the weather for in YYYY-MM-DD format"`
	}

	// GeoResponse는 OpenWeatherMap Geocoding API 응답 구조체
	GeoResponse struct {
		Lat  float64 `json:"lat"`
		Lon  float64 `json:"lon"`
		Name string  `json:"name"`
	}

	// WeatherSummaryResponse는 OpenWeatherMap One Call API 3.0 `/onecall/day_summary` 응답 구조체
	GetWeatherResponse struct {
		Humidity struct {
			Afternoon float64 `json:"afternoon"`
		} `json:"humidity"`
		Temperature struct {
			Min       float64 `json:"min"`
			Max       float64 `json:"max"`
			Afternoon float64 `json:"afternoon"`
			Night     float64 `json:"night"`
			Evening   float64 `json:"evening"`
			Morning   float64 `json:"morning"`
		} `json:"temperature"`
		Wind struct {
			Max struct {
				Speed     float64 `json:"speed"`
				Direction float64 `json:"direction"`
			} `json:"max"`
		} `json:"wind"`
	}

	// APIErrorResponse는 API 호출 실패 시 반환되는 JSON 구조체
	APIErrorResponse struct {
		Code       int      `json:"cod"`
		Message    string   `json:"message"`
		Parameters []string `json:"parameters"`
	}
)

// getCoordinates: 도시명을 위도/경도로 변환
func getCoordinates(apiKey string, city string) (float64, float64, error) {
	baseURL := "http://api.openweathermap.org/geo/1.0/direct"
	params := url.Values{}
	params.Set("q", city)
	params.Set("limit", "1")
	params.Set("appid", apiKey)

	reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

	resp, err := http.Get(reqURL)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("위도/경도 변환 API 호출 실패: %s", resp.Status)
	}

	var geoData []GeoResponse
	if err := json.NewDecoder(resp.Body).Decode(&geoData); err != nil {
		return 0, 0, err
	}

	if len(geoData) == 0 {
		return 0, 0, fmt.Errorf("도시명을 찾을 수 없습니다: %s", city)
	}

	return geoData[0].Lat, geoData[0].Lon, nil
}

// getWeatherSummary: `/onecall/day_summary` API 호출하여 특정 날짜의 날씨 요약 가져오기
func getWeatherSummary(apiKey string, date string, latitude, longitude float64, unit, lang string) (*GetWeatherResponse, error) {
	baseURL := "https://api.openweathermap.org/data/3.0/onecall/day_summary"
	params := url.Values{}
	params.Set("lat", fmt.Sprintf("%f", latitude))
	params.Set("lon", fmt.Sprintf("%f", longitude))
	params.Set("date", date)
	params.Set("appid", apiKey)
	params.Set("unit", unit)
	params.Set("lang", lang)

	reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

	resp, err := http.Get(reqURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// API 에러 응답 처리
	if resp.StatusCode != http.StatusOK {
		var apiErr APIErrorResponse
		if err := json.NewDecoder(resp.Body).Decode(&apiErr); err != nil {
			return nil, fmt.Errorf("API 호출 실패: HTTP %d (응답 해석 실패)", resp.StatusCode)
		}
		return nil, fmt.Errorf("API 호출 실패: HTTP %d, 메시지: %s, 매개변수: %v", apiErr.Code, apiErr.Message, apiErr.Parameters)
	}

	var weatherResp GetWeatherResponse
	if err := json.NewDecoder(resp.Body).Decode(&weatherResp); err != nil {
		return nil, err
	}

	return &weatherResp, nil
}

func (m *manager) GetWeather(ctx context.Context, req *GetWeatherRequest, apiKey string) (*GetWeatherResponse, error) {
	if strings.Contains(req.Location, "HKCEC") {
		req.Location = "HK"
	}
	m.logger.Debug("get_weather", "location", req.Location, "date", req.Date)

	latitude, longitude, err := getCoordinates(apiKey, req.Location)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to convert coordinates")
	}

	weatherSummary, err := getWeatherSummary(apiKey, req.Date, latitude, longitude, "metric", "en")
	if err != nil {
		return nil, errors.Wrapf(err, "error occurred while fetching weather information")
	}

	return weatherSummary, nil
}

func (m *manager) registerGetWeatherTool(skill *entity.NativeAgentSkill) error {
	return registerNativeTool(
		m,
		"get_weather",
		"Get weather information when you need it",
		skill,
		func(ctx *Context, req struct {
			*GetWeatherRequest
		}) (res struct {
			*GetWeatherResponse
		}, err error) {
			res.GetWeatherResponse, err = m.GetWeather(ctx, req.GetWeatherRequest, skill.Env["OPENWEATHER_API_KEY"].(string))
			return
		},
	)
}
